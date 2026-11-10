// sigprof.c: a SIGPROF sampler to preload (round sp53; the host has no perf without root).
//   cc -O2 -fPIC -shared -o sigprof.so sigprof.c
//   SIGPROF_OUT=/path/out.txt [SIGPROF_HZ=1000] taskset -c 6 env LD_PRELOAD=/path/sigprof.so <program>
// (preload it after `taskset env`, not on taskset itself: the output of the wrong process is overwritten otherwise.)
// Every tick of ITIMER_PROF (process CPU time) records the interrupted instruction's address. At exit the samples are written as
// "<module> <offset in the file, hex> <count>" lines, sorted by module and offset, after a "# samples N" line; addresses in no file mapping are "[anon]".
// Symbolise with sigprof_sym.py (nm / addr2line on the modules).
#define _GNU_SOURCE
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <ucontext.h>
#include <unistd.h>

#define CAP (1u << 22)
static unsigned long* g_buf;
static volatile unsigned g_n;
static pid_t g_pid;

static void on_prof(int sig, siginfo_t* si, void* uc_) {
  (void)sig; (void)si;
  ucontext_t* uc = (ucontext_t*)uc_;
  unsigned i = __atomic_fetch_add(&g_n, 1, __ATOMIC_RELAXED);
  if (i < CAP) g_buf[i] = (unsigned long)uc->uc_mcontext.gregs[REG_RIP];
}

struct map { unsigned long lo, hi, off; char path[256]; };
static int cmp_ul(const void* a, const void* b) { unsigned long x = *(const unsigned long*)a, y = *(const unsigned long*)b; return x < y ? -1 : x > y; }

static void dump(void) {
  if (getpid() != g_pid) return;   // (a forked child: not ours to write)
  struct itimerval z = {{0, 0}, {0, 0}};
  setitimer(ITIMER_PROF, &z, 0);
  const char* out = getenv("SIGPROF_OUT");
  if (!out) return;
  unsigned n = g_n < CAP ? g_n : CAP;
  static struct map maps[4096];
  unsigned nm = 0;
  FILE* f = fopen("/proc/self/maps", "r");
  char line[1024];
  while (f && fgets(line, sizeof line, f) && nm < 4096) {
    unsigned long lo, hi, off; char perms[8], path[512] = "";
    if (sscanf(line, "%lx-%lx %7s %lx %*s %*s %511[^\n]", &lo, &hi, perms, &off, path) < 4) continue;
    if (perms[2] != 'x') continue;
    maps[nm].lo = lo; maps[nm].hi = hi; maps[nm].off = off;
    const char* p = path; while (*p == ' ') ++p;
    snprintf(maps[nm].path, sizeof maps[nm].path, "%s", *p == '/' ? p : "[anon]");
    ++nm;
  }
  if (f) fclose(f);
  qsort(g_buf, n, sizeof *g_buf, cmp_ul);
  FILE* o = fopen(out, "w");
  if (!o) return;
  fprintf(o, "# samples %u\n", n);
  for (unsigned i = 0; i < n;) {
    unsigned j = i;
    while (j < n && g_buf[j] == g_buf[i]) ++j;
    unsigned long a = g_buf[i];
    const char* path = "[unmapped]"; unsigned long rel = a;
    for (unsigned m = 0; m < nm; ++m)
      if (a >= maps[m].lo && a < maps[m].hi) { path = maps[m].path; rel = a - maps[m].lo + maps[m].off; break; }
    fprintf(o, "%s %lx %u\n", path, rel, j - i);
    i = j;
  }
  fclose(o);
}

__attribute__((constructor)) static void init(void) {
  g_pid = getpid();
  g_buf = (unsigned long*)malloc(CAP * sizeof *g_buf);
  if (!g_buf) return;
  struct sigaction sa;
  memset(&sa, 0, sizeof sa);
  sa.sa_sigaction = on_prof;
  sa.sa_flags = SA_SIGINFO | SA_RESTART;
  sigaction(SIGPROF, &sa, 0);
  unsigned hz = getenv("SIGPROF_HZ") ? (unsigned)atoi(getenv("SIGPROF_HZ")) : 1000;
  if (!hz) hz = 1000;
  struct itimerval it = {{0, 1000000 / hz}, {0, 1000000 / hz}};
  setitimer(ITIMER_PROF, &it, 0);
  atexit(dump);
}
