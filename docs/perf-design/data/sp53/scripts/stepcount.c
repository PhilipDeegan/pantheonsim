// stepcount.c: exact instruction counts of a fixed piece of a JIT run, on a host without perf (round sp53).
//   cc -O2 -fPIC -shared -o stepcount.so stepcount.c -ldl
//   STEP_OUT=out.txt STEP_SKIP=96 STEP_CALLS=48 taskset -c 6 env LD_PRELOAD=/path/stepcount.so <program>     (warm JIT cache, VGPU_THREADS=1)
// The library interposes dlopen. When the simulator loads a generated object (<32 hex digits>.so) it puts a breakpoint at every region function
// (vgpu_jit_r<n>) and lets STEP_SKIP region calls go by; from the next one on the thread runs with the trap flag set, one SIGTRAP an instruction, and
// every instruction executed is counted by its address until STEP_CALLS more region calls have begun. Then the counts are written as
// "<module> <offset> <count>" lines (the format of sigprof.c: sigprof_diff.py and sigprof_addr.py read it) and the process ends.
// Choose STEP_SKIP and STEP_CALLS as whole blocks of the kernel's region calls (SYR2K: 24 a block without linking, 16 with), so that two settings
// are compared over exactly the same simulated work. About a microsecond an instruction.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <ucontext.h>
#include <unistd.h>

#define NFN 16
#define TAB (1u << 20)
static uintptr_t g_fn[NFN];
static unsigned char g_orig[NFN];
static unsigned g_nfn;
static long g_skip, g_calls, g_seen, g_measured;
static int g_state;   // 0 armed (breakpoints in), 1 one step over a restored breakpoint (g_re), 2 counting
static unsigned g_re;
static struct { uintptr_t rip; uint64_t n; } g_tab[TAB];
static uint64_t g_steps;

static void poke(uintptr_t a, unsigned char b) {
  uintptr_t page = a & ~(uintptr_t)4095;
  mprotect((void*)page, 4096, PROT_READ | PROT_WRITE | PROT_EXEC);
  *(unsigned char*)a = b;
}
static void wr(int fd, const char* s) { if (write(fd, s, strlen(s)) < 0) {} }
static void hex(char* b, uint64_t v) { char t[20]; int n = 0; do { t[n++] = "0123456789abcdef"[v & 15]; v >>= 4; } while (v); while (n) *b++ = t[--n]; *b = 0; }
static void dec(char* b, uint64_t v) { char t[24]; int n = 0; do { t[n++] = (char)('0' + v % 10); v /= 10; } while (v); while (n) *b++ = t[--n]; *b = 0; }

static char g_maps[1 << 19];
struct map { uintptr_t lo, hi, off; const char* path; };
static struct map g_m[2048];

static void dump_and_exit(void) {
  const char* out = getenv("STEP_OUT");
  int fd = open(out ? out : "/dev/null", O_WRONLY | O_CREAT | O_TRUNC, 0644);
  int mf = open("/proc/self/maps", O_RDONLY);
  size_t len = 0; ssize_t r;
  while (mf >= 0 && len < sizeof g_maps - 1 && (r = read(mf, g_maps + len, sizeof g_maps - 1 - len)) > 0) len += (size_t)r;
  g_maps[len] = 0;
  unsigned nm = 0;
  for (char* l = g_maps; *l && nm < 2048;) {
    char* e = strchr(l, '\n'); if (e) *e = 0;
    unsigned long lo, hi, off; char perms[8]; int pos = 0;
    if (sscanf(l, "%lx-%lx %7s %lx %*s %*s %n", &lo, &hi, perms, &off, &pos) >= 4 && perms[2] == 'x') {
      g_m[nm].lo = lo; g_m[nm].hi = hi; g_m[nm].off = off; g_m[nm].path = (pos && l[pos] == '/') ? l + pos : "[anon]"; ++nm;
    }
    if (!e) break; l = e + 1;
  }
  char b[64];
  wr(fd, "# samples "); dec(b, g_steps); wr(fd, b); wr(fd, "\n");
  for (unsigned i = 0; i < TAB; ++i) {
    if (!g_tab[i].n) continue;
    uintptr_t a = g_tab[i].rip; const char* path = "[unmapped]"; uintptr_t rel = a;
    for (unsigned m = 0; m < nm; ++m) if (a >= g_m[m].lo && a < g_m[m].hi) { path = g_m[m].path; rel = a - g_m[m].lo + g_m[m].off; break; }
    wr(fd, path); wr(fd, " "); hex(b, rel); wr(fd, b); wr(fd, " "); dec(b, g_tab[i].n); wr(fd, b); wr(fd, "\n");
  }
  close(fd);
  _exit(0);
}

static void on_trap(int sig, siginfo_t* si, void* uc_) {
  (void)sig; (void)si;
  ucontext_t* uc = (ucontext_t*)uc_;
  greg_t* g = uc->uc_mcontext.gregs;
  uintptr_t rip = (uintptr_t)g[REG_RIP];
  if (g_state == 2) {   // rip: the instruction about to run
    for (unsigned i = 0; i < g_nfn; ++i) if (rip == g_fn[i]) { if (++g_measured > g_calls) dump_and_exit(); break; }
    ++g_steps;
    uint32_t h = (uint32_t)((rip * 0x9E3779B97F4A7C15ull) >> 44) & (TAB - 1);
    while (g_tab[h].n && g_tab[h].rip != rip) h = (h + 1) & (TAB - 1);
    g_tab[h].rip = rip; ++g_tab[h].n;
    return;   // (the trap flag stays in the saved flags)
  }
  if (g_state == 1) {   // one instruction past a restored breakpoint: put it back, run on
    poke(g_fn[g_re], 0xCC);
    g[REG_EFL] &= ~0x100L;
    g_state = 0;
    return;
  }
  for (unsigned i = 0; i < g_nfn; ++i) {
    if (rip - 1 != g_fn[i]) continue;
    g[REG_RIP] = (greg_t)(rip - 1);
    g[REG_EFL] |= 0x100L;
    if (++g_seen > g_skip) {   // from here on: every instruction
      for (unsigned k = 0; k < g_nfn; ++k) poke(g_fn[k], g_orig[k]);
      g_state = 2; g_measured = 1; ++g_steps;
      uint32_t h = (uint32_t)(((rip - 1) * 0x9E3779B97F4A7C15ull) >> 44) & (TAB - 1);
      g_tab[h].rip = rip - 1; ++g_tab[h].n;
    } else {
      poke(g_fn[i], g_orig[i]); g_re = i; g_state = 1;
    }
    return;
  }
}

void* dlopen(const char* path, int flags) {
  static void* (*real)(const char*, int);
  if (!real) real = (void* (*)(const char*, int))dlsym(RTLD_NEXT, "dlopen");
  void* h = real(path, flags);
  size_t n = path ? strlen(path) : 0;
  if (!h || n < 35 || strcmp(path + n - 3, ".so") || strspn(path + n - 35, "0123456789abcdef") != 32 || path[n - 36] != '/') return h;
  if (!getenv("STEP_OUT")) return h;
  if (!g_nfn) {
    g_skip = getenv("STEP_SKIP") ? atol(getenv("STEP_SKIP")) : 0;
    g_calls = getenv("STEP_CALLS") ? atol(getenv("STEP_CALLS")) : 16;
    struct sigaction sa; memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = on_trap; sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigaction(SIGTRAP, &sa, 0);
  }
  for (unsigned i = 0; i < 64 && g_nfn < NFN; ++i) {
    char name[32]; snprintf(name, sizeof name, "vgpu_jit_r%u", i);
    void* f = dlsym(h, name);
    if (!f) break;
    g_fn[g_nfn] = (uintptr_t)f; g_orig[g_nfn] = *(unsigned char*)f; poke((uintptr_t)f, 0xCC); ++g_nfn;
  }
  return h;
}
