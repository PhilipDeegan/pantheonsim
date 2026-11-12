// Host-GPU mode, `exact` (docs/perf-design/09-host-gpu-spike.md): a launch whose kernel passes the eligibility scan and whose
// size rule says the card path wins runs on the machine's real GPU, on the simulator's own device memory.
// `relaxed` (docs/perf-design/12-host-gpu-relaxed.md) is the same machinery with a wider eligibility scan (kernels whose result is the card's, not the
// simulator's bits) and a verify that compares by tolerance; every safety check, the size rule and the residency protocol are shared.
//
// How, in this first form (write-through, no state between launches): the simulator's device memory stays the truth. For a launch
// every live allocation is copied to the card at the same virtual address (the driver's virtual-memory calls reserve the
// simulator's device window and back the granules in use), the kernel runs there from the module's own cubin, and what changed
// is copied back into the simulator. A launch that fails on the card changes nothing and runs in the simulator after all.
// The real driver is loaded next to the shim with dlopen (RTLD_DEEPBIND: its symbols are its own), from a thread that is given
// a deadline, because a reservation outside the card's address range can hang instead of failing (WSL2).
#include "vgpu/host_gpu_rt.hpp"

#include <dlfcn.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <set>
#include <thread>

#include "vgpu/error.hpp"
#include "vgpu/host_gpu_track.hpp"
#include "vgpu/sass/exec.hpp"

namespace vgpu::hostgpu {
namespace {

// The few driver types and calls used, declared here so that this file needs no CUDA header (the simulator builds without one).
using CUresult = int;
using CUdevice = int;
using CUcontext = struct CUctx_st*;
using CUmodule = struct CUmod_st*;
using CUfunction = struct CUfunc_st*;
using CUdeviceptr = unsigned long long;
using CUhandle = unsigned long long;

struct CUmemLocation { int type; int id; };
struct CUmemAllocationProp {
  int type;
  int requestedHandleTypes;
  CUmemLocation location;
  void* win32HandleMetaData;
  struct { unsigned char compressionType; unsigned char gpuDirectRDMACapable; unsigned short usage; unsigned char reserved[4]; } allocFlags;
};
struct CUmemAccessDesc { CUmemLocation location; int flags; };

constexpr uint64_t kGranule = 2ull << 20;
constexpr int kAttrMultiprocessors = 16, kAttrCcMajor = 75, kAttrCcMinor = 76;
constexpr int kFuncAttrMaxDynamicShared = 8;

struct Api {
  void* lib = nullptr;
  CUresult (*cuInit)(unsigned) = nullptr;
  CUresult (*cuDeviceGet)(CUdevice*, int) = nullptr;
  CUresult (*cuDeviceGetAttribute)(int*, int, CUdevice) = nullptr;
  CUresult (*cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice) = nullptr;
  CUresult (*cuCtxSetCurrent)(CUcontext) = nullptr;
  CUresult (*cuCtxSynchronize)() = nullptr;
  CUresult (*cuMemAddressReserve)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long) = nullptr;
  CUresult (*cuMemCreate)(CUhandle*, size_t, const CUmemAllocationProp*, unsigned long long) = nullptr;
  CUresult (*cuMemMap)(CUdeviceptr, size_t, size_t, CUhandle, unsigned long long) = nullptr;
  CUresult (*cuMemSetAccess)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t) = nullptr;
  CUresult (*cuMemsetD8)(CUdeviceptr, unsigned char, size_t) = nullptr;
  CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void*, size_t) = nullptr;
  CUresult (*cuMemcpyDtoH)(void*, CUdeviceptr, size_t) = nullptr;
  CUresult (*cuMemGetInfo)(size_t*, size_t*) = nullptr;
  CUresult (*cuModuleLoadData)(CUmodule*, const void*) = nullptr;
  CUresult (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
  CUresult (*cuFuncSetAttribute)(CUfunction, int, int) = nullptr;
  CUresult (*cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void*, void**, void**) = nullptr;

  bool load(const char* path) {
    lib = dlopen(path, RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    if (!lib) return false;
    // The names the driver exports: cuda.h renames these calls to their _v2 versions with a macro, and the plain names are the
    // old 32-bit-size ones.
#define VGPU_HG_SYM(n) \
  if (!(n = reinterpret_cast<decltype(n)>(dlsym(lib, #n)))) return false;
#define VGPU_HG_SYM_V2(n) \
  if (!(n = reinterpret_cast<decltype(n)>(dlsym(lib, #n "_v2")))) return false;
    VGPU_HG_SYM(cuInit) VGPU_HG_SYM(cuDeviceGet) VGPU_HG_SYM(cuDeviceGetAttribute) VGPU_HG_SYM(cuDevicePrimaryCtxRetain)
    VGPU_HG_SYM(cuCtxSetCurrent) VGPU_HG_SYM(cuCtxSynchronize) VGPU_HG_SYM(cuMemAddressReserve) VGPU_HG_SYM(cuMemCreate)
    VGPU_HG_SYM(cuMemMap) VGPU_HG_SYM(cuMemSetAccess) VGPU_HG_SYM_V2(cuMemsetD8) VGPU_HG_SYM_V2(cuMemcpyHtoD) VGPU_HG_SYM_V2(cuMemcpyDtoH)
    VGPU_HG_SYM_V2(cuMemGetInfo) VGPU_HG_SYM(cuModuleLoadData) VGPU_HG_SYM(cuModuleGetFunction) VGPU_HG_SYM(cuFuncSetAttribute)
    VGPU_HG_SYM(cuLaunchKernel)
#undef VGPU_HG_SYM
#undef VGPU_HG_SYM_V2
    return true;
  }
};

// VGPU_HOST_GPU_TRACE=<file> (spike, step A): the stream of allocations, frees, host copies and launches, for an offline model of the transfers
// a persistent card copy would still need (docs/perf-design/data/round13/residency_model.py). Events the mode itself causes are not recorded.
thread_local bool t_internal = false;
FILE* g_trace_file = nullptr;
std::mutex g_trace_mu;
void trace_hook(char op, uint64_t a, uint64_t b) {
  if (t_internal || !g_trace_file) return;
  std::lock_guard<std::mutex> g(g_trace_mu);
  std::fprintf(g_trace_file, "%c %llx %llx\n", op, static_cast<unsigned long long>(a), static_cast<unsigned long long>(b));
}
// Installed when the library loads, so the allocations and copies before the first launch are in the trace.
struct TraceInit {
  TraceInit() {
    const char* tf = std::getenv("VGPU_HOST_GPU_TRACE");
    if (!tf || !*tf) return;
    g_trace_file = std::fopen(tf, "w");
    if (!g_trace_file) return;
    std::setvbuf(g_trace_file, nullptr, _IOFBF, 1 << 20);
    set_mem_trace(trace_hook);
    std::atexit([] { std::lock_guard<std::mutex> g(g_trace_mu); std::fflush(g_trace_file); });
  }
} g_trace_init;
struct InternalScope {
  bool prev;
  InternalScope() : prev(t_internal) { t_internal = true; }
  ~InternalScope() { t_internal = prev; }
};

// What launch() learned of the launch this thread is about to hand to the simulator, for sample_plan and sample_decide (which the same thread calls next).
struct SampleHint {
  bool set = false;
  uint64_t blocks = 0, live = 0, instrs = 0;
  uint32_t threads_per_block = 0;
  bool back_edge = false;
  double static_s = 0;       // the static estimate (static_sim_seconds)
  unsigned samples = 0;
  double age_s = 0;
  bool affordable = true;    // whether the static floor or the process's age allows the sample (log and always sample without asking)
};
thread_local SampleHint t_sample_hint;
// The sample of the launch this thread's simulator is about to run whole, for the log line note_simulated prints (what it said, what the launch took).
struct LastSample {
  bool set = false;
  SampleHint hint;
  Sample sample{};
  uint64_t ran = 0;
  double first_block_s = 0, wall_s = 0, fork_s = 0, estimate_s = 0;
  bool answered = false, would_send = false;
};
thread_local LastSample t_last_sample;
const std::chrono::steady_clock::time_point g_loaded = std::chrono::steady_clock::now();   // (the library's load: the process's age is counted from it)

bool quiet() {
  const char* q = std::getenv("VGPU_QUIET");
  return q && *q && *q != '0';
}
bool logging() {
  const char* q = std::getenv("VGPU_HOST_GPU_LOG");
  return q && *q && *q != '0';
}
// VGPU_HOST_GPU_LOG=2: also the SASS text of a kernel the exact set refuses for an SFU instruction (to read what the instruction is part of).
bool logging_code() {
  const char* q = std::getenv("VGPU_HOST_GPU_LOG");
  return q && std::string(q) == "2";
}

}  // namespace

struct Runtime::Impl : CardGate {
  Api api;
  CUcontext ctx = nullptr;
  uint64_t window_base = 0;
  uint64_t mapped_end = 0;   // granules [window_base, mapped_end) are backed
  Stats stats;
  Costs costs;
  SimTimeTable sim_times;
  std::map<std::string, double> card_kernel_s;
  TrafficTable traffic;
  std::set<std::string> banned;   // kernels whose offloaded launch took longer than the simulator was expected to
  std::map<const sass::Module*, CUmodule> modules;
  // Why a launch stays in the simulator before the size rule is asked: a safety check (the launch's shape, the module, memory the card does not hold),
  // or the kernel's code.
  enum class Refused : uint8_t { No, Safety, Code };
  struct Verdict {
    bool ok = false;
    uint64_t instrs = 0;
    bool runnable = false;   // everything but the instruction scan passed: the card could run it
    bool back_edge = false;  // the code has a loop (a branch to itself or an earlier instruction)
    Refused kind = Refused::No;
    RelaxedRefusal rr = RelaxedRefusal::None;   // Code: which class of instruction (in exact mode: what relaxed mode would say with its defaults)
    const char* why = "";                       // the reason given on a later launch of the same kernel
  };
  Mode mode = Mode::Exact;
  RelaxedSwitches sw;      // relaxed mode's switches; the defaults in exact mode (where they only class the refusals for the summary)
  Tolerance tol;           // relaxed mode's verify (VGPU_HOST_GPU_ATOL, VGPU_HOST_GPU_RTOL)
  struct KernelTol {
    TolReport rep;
    unsigned said = 0;     // launches outside tolerance already reported without VGPU_HOST_GPU_LOG
  };
  std::map<std::string, KernelTol> tol_by_kernel;

  // VGPU_HOST_GPU_TIMEOUT (seconds; relaxed mode: 600 unless set, exact mode: none unless set). cuCtxSynchronize has no deadline, a kernel on the card
  // cannot be cancelled, and with a kernel still running the card's memory cannot be read back, so there is nothing to fall back to: when a launch has
  // been on the card for longer than the limit the process says so and ends, instead of hanging without a word. Exact mode had no such launch to fear
  // beyond the program's own (the card computes the simulator's bits, so it loops exactly when the simulator would); in relaxed mode a loop that waits
  // on a value the card computes differently, or on a lock another thread never gives back, is the card's to run.
  struct Watchdog {
    std::mutex mu;
    std::condition_variable cv;
    double limit_s = 0;
    bool started = false, armed = false;
    std::chrono::steady_clock::time_point deadline{};
    std::string kernel;
    void arm(const std::string& k) {
      if (limit_s <= 0) return;
      std::lock_guard<std::mutex> g(mu);
      kernel = k;
      deadline = std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(limit_s));
      armed = true;
      if (!started) {
        started = true;
        std::thread([this] { run(); }).detach();
      }
      cv.notify_all();
    }
    void disarm() {
      if (limit_s <= 0) return;
      std::lock_guard<std::mutex> g(mu);
      armed = false;
      cv.notify_all();
    }
    void run() {
      std::unique_lock<std::mutex> lk(mu);
      for (;;) {
        if (!armed) {
          cv.wait(lk);
          continue;
        }
        const auto until = deadline;
        cv.wait_until(lk, until);
        if (armed && deadline == until && std::chrono::steady_clock::now() >= until) {
          std::fprintf(stderr,
                       "host_gpu: %s has been on the card for more than %g s (VGPU_HOST_GPU_TIMEOUT): a kernel on the card cannot be cancelled and its memory cannot be read back; the process ends\n",
                       kernel.c_str(), limit_s);
          std::fflush(nullptr);
          _exit(124);
        }
      }
    }
  } watchdog;
  std::map<std::pair<const sass::Module*, std::string>, Verdict> eligible;
  GateLock mu;   // shared: the gate's accesses that change nothing (before_access); exclusive: everything else
  uint64_t max_bytes = 2ull << 30;
  bool dead = false;
  uint32_t multiprocessors = 0;   // of the simulated profile (and the card: start() checks they agree)
  // The card copy outlives a launch (docs/perf-design/09-host-gpu-spike.md section 4) unless VGPU_HOST_GPU_WRITE_THROUGH=1.
  bool resident = true;
  MemoryManager* gate_mem = nullptr;
  using Res = Tracked;   // (vgpu/host_gpu_track.hpp)
  TrackedMap res;
  // Scratch for the copies between the card and the simulator, kept across launches: a fresh 16 to 32 MiB vector per launch is an mmap, a zero fill and a
  // page fault per page each time (4.7 s of system time in 1,400 launches of PolyBench LU, against 0.6 s with the write-through path).
  std::vector<uint8_t> pull_buf, push_buf, sim_buf;
  bool verify_card_ran = false;   // verify + resident: this launch ran on the card and the simulator runs it next, on memory that is the truth
  struct Last {
    bool in_set = false;
    double card_s = -1;
  } last;
  // The simulator time spent so far in launches of exact-set kernels that stayed in the simulator: the ski-rental account of should_offload_with_start (host_gpu.hpp).
  double eligible_sim_spent_s = 0;
  // VGPU_HOST_GPU_VERIFY=1 runs the launches the mode would offload on the card and then in the simulator, and compares the memory
  // (the simulator's result is the one the program gets); =all does it for every kernel the card can run, in or out of the exact set.
  int verify = 0;
  // VGPU_HOST_GPU_STATIC=1: a kernel the simulator has not run is estimated from its code and grid. NOT a lower bound (threads that leave early
  // execute less: PolyBench 3DCONV was 38x slower with it), so it is an experiment switch, off by default.
  bool static_estimate = false;
  // VGPU_HOST_GPU_NO_FIRST_LAUNCH=1: the first launch of a kernel is always the simulator's (the behaviour before the first-launch rule; for A/B runs).
  bool first_launch_off = false;
  // ---- a sampled launch (vgpu/host_gpu.hpp "The first blocks of a launch, timed") ----
  // VGPU_HOST_GPU_SAMPLE: 0 samples nothing (the rule before sp60b). `log` samples and decides but sends no launch for it, and prints one
  // host_gpu_sample line a sampled launch (what was measured, what would have been done, what the launch then took): a model of the rule with no card.
  // `always` samples whatever the process's age (sample_affordable is not asked), for tests and experiments; `log` does not ask it either and
  // prints what it would have said.
  enum class SampleMode : uint8_t { On, Off, Log, Always } sample_mode = SampleMode::On;
  double sample_spent_s = 0;      // what the samples of this process have cost
  double sample_dearest_s = 0;    // the dearest of them
  // VGPU_HOST_GPU_EARLY_START (an experiment of Round sp60b, off by default): 1 starts the driver, on its thread, at the first launch of an admitted
  // kernel with a loop on a grid that fills the card, whether or not that launch goes: later decisions then find the start paid. What it costs a
  // program that never uses the driver is the wait for the start at exit; 2 does not wait at exit (to measure whether a process can end while the
  // driver starts: nothing says it can).
  int early_start = 0;
  struct Decision;
  bool launch(const ptx::EntryFn& fn, const exec::LaunchConfig& cfg, const std::vector<std::vector<uint8_t>>& args, MemoryManager& mem, const double* forced_sim_s);
  void consider_sample(const ptx::EntryFn& fn, const exec::LaunchConfig& cfg, const Decision& dec, uint64_t blocks, MemoryManager& mem);

  struct Pending {
    bool active = false, in_exact_set = false;
    std::string kernel;
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    std::vector<std::vector<uint8_t>> card;
  } pending;


  // ---- the gate (vgpu::CardGate): called by the memory manager on the slow path of any access to an allocation with a valid card copy ----
  static uint64_t chunk_len(const Res& a, uint64_t c) { return std::min<uint64_t>(kChunkSize, a.size - c * kChunkSize); }

  void update_slot(Res& a) {
    const uint32_t v = static_cast<uint32_t>(a.r.card_valid());
    const uint32_t old = a.slot->exchange(v, std::memory_order_relaxed);
    if (old == 0 && v > 0) gate_mem->card_resident_count().fetch_add(1, std::memory_order_relaxed);
    if (old > 0 && v == 0) gate_mem->card_resident_count().fetch_sub(1, std::memory_order_relaxed);
  }

  // Copies chunks the card alone holds into the simulator's memory (not in a verify run, where the simulator's memory is the truth already).
  void pull_chunks(Res& a, const std::vector<uint64_t>& idx) {
    if (idx.empty() || verify) return;
    api.cuCtxSetCurrent(ctx);
    std::vector<uint8_t>& buf = pull_buf;
    for (const auto& [first, count] : chunk_runs(idx)) {
      for (uint64_t c = first; c < first + count;) {
        const uint64_t n_chunks = std::min<uint64_t>(first + count - c, (32ull << 20) / kChunkSize);
        const uint64_t off = c * kChunkSize, bytes = std::min<uint64_t>(n_chunks * kChunkSize, a.size - off);
        buf.resize(bytes);
        if (api.cuMemcpyDtoH(buf.data(), a.base + off, bytes)) {
          dead = true;
          throw Error(Err::Internal, "host_gpu: reading the card's memory back failed: data only the card held is lost");
        }
        gate_mem->write(a.base + off, buf.data(), bytes);
        stats.pulled_bytes += bytes;
        c += n_chunks;
      }
    }
  }

  void before_access(uint64_t addr, uint64_t len, bool write) override {
    if (t_internal) return;   // the mode's own copies
    // (vgpu/host_gpu_track.hpp: the usual access changes nothing and takes only the shared lock)
    gate_access(mu, res, addr, len, write,
                [&](Res& a, const std::vector<uint64_t>& idx) { InternalScope internal; pull_chunks(a, idx); },
                [&](Res& a) { update_slot(a); });
  }

  void before_free(uint64_t base) override {
    if (t_internal) return;
    std::lock_guard<GateLock> g(mu);
    for (auto it = base == UINT64_MAX ? res.begin() : res.find(base); it != res.end();) {
      Res& a = it->second;
      a.r = Residency(a.r.chunks());   // dropped: nothing of it is valid on the card any more
      update_slot(a);
      if (base == UINT64_MAX) it = res.erase(it);
      else { res.erase(it); break; }
    }
  }

  // The simulator is about to run a launch of its own (mu held by the caller).
  void reset_for_simulator() {
    InternalScope internal;
    for (auto& [b, a] : res) {
      if (a.slot->load(std::memory_order_relaxed) == 0) continue;
      pull_chunks(a, a.r.sim_launch());
      update_slot(a);
    }
  }

  // The tracked allocations are the live ones: new ones start Host (nothing of them is on the card), the ones that went are dropped (sync_tracked).
  void sync_tracking(MemoryManager& mem, const std::vector<std::pair<uint64_t, uint64_t>>& ranges) {
    if (gate_mem != &mem) {
      gate_mem = &mem;
      mem.set_card_gate(this);
    }
    sync_tracked(res, mem, ranges);
  }

  // ---- start-up of the real driver: lazy ----
  // Loading the driver, cuInit, the primary context and the 1 TiB reservation cost 0.15 to 0.35 s (09 doc, section 5; 0.23 s on an A10G, Round sp42). Nothing happens
  // until try_launch has decided that a launch goes to the card (should_offload_with_start charges this cost in that decision): `await_start` is the only caller of
  // `begin_start` (a thread with a deadline: everything that can hang runs on one) and waits for it. A program that never decides on the card never loads the driver,
  // starts no thread and has nothing to wait for at exit. (Before sp43 the thread was started at the first launch of any exact-set kernel on 32 warps or more,
  // whether or not the launch went to the card, and `settle_start` made the process wait for it at exit.)
  struct StartShared {
    std::mutex mu;
    std::condition_variable cv;
    bool done = false, ok = false;
    std::string why;
  };
  std::shared_ptr<StartShared> start_sh;     // set once the thread is launched
  std::chrono::steady_clock::time_point start_began{};
  bool start_failed = false;                 // the driver could not be used (said once); the mode is off for the rest of the process
  bool start_ready = false;                  // the driver is up: window_base and mapped_end are set
  int want_sm = 0;
  uint32_t want_mp = 0;
  uint64_t va_base = 0;
  std::string lib_path;

  void configure_start(const DeviceProfile& profile, uint64_t base) {
    const char* path = std::getenv("VGPU_HOST_GPU_LIBCUDA");
    lib_path = path && *path ? path : "/usr/lib/x86_64-linux-gnu/libcuda.so.1";
    want_sm = profile.cc_major * 10 + profile.cc_minor;
    want_mp = profile.limits.multiprocessors;
    multiprocessors = want_mp;
    va_base = base;
  }

  // Starts the driver on a thread and returns at once (idempotent). The thread only touches `api` and `ctx`, which nobody reads before await_start.
  void begin_start() {
    if (start_sh || start_failed) return;
    auto sh = std::make_shared<StartShared>();
    start_sh = sh;
    start_began = std::chrono::steady_clock::now();
    Impl* self = this;
    const std::string lib = lib_path;
    const int sm = want_sm;
    const uint32_t mp_want = want_mp;
    const uint64_t base = va_base;
    std::thread([sh, self, lib, sm, mp_want, base] {
      bool ok = false;
      std::string why;
      do {
        if (!self->api.load(lib.c_str())) { why = "cannot load the real driver " + lib; break; }
        if (self->api.cuInit(0)) { why = "cuInit failed"; break; }
        CUdevice d;
        if (self->api.cuDeviceGet(&d, 0)) { why = "no device 0"; break; }
        int mp = 0, cmaj = 0, cmin = 0;
        self->api.cuDeviceGetAttribute(&mp, kAttrMultiprocessors, d);
        self->api.cuDeviceGetAttribute(&cmaj, kAttrCcMajor, d);
        self->api.cuDeviceGetAttribute(&cmin, kAttrCcMinor, d);
        if (cmaj * 10 + cmin != sm || static_cast<uint32_t>(mp) != mp_want) {
          why = "the card (sm_" + std::to_string(cmaj * 10 + cmin) + ", " + std::to_string(mp) + " SMs) is not the simulated device (sm_" +
                std::to_string(sm) + ", " + std::to_string(mp_want) + " SMs)";
          break;
        }
        if (self->api.cuDevicePrimaryCtxRetain(&self->ctx, d) || self->api.cuCtxSetCurrent(self->ctx)) { why = "no context"; break; }
        CUdeviceptr p = 0;
        const CUresult r = self->api.cuMemAddressReserve(&p, kDeviceVaStride, 0, base, 0);
        if (r || p != base) { why = "the card did not give the simulator's address window"; break; }
        ok = true;
      } while (false);
      std::lock_guard<std::mutex> g(sh->mu);
      sh->done = true;
      sh->ok = ok;
      sh->why = why;
      sh->cv.notify_all();
    }).detach();
  }

  // Waits (to the 10 s deadline, counted from the start) for the driver. False: it cannot be used; the mode is off from now on.
  bool await_start() {
    if (start_ready) return true;
    if (start_failed) return false;
    begin_start();
    auto sh = start_sh;
    std::unique_lock<std::mutex> lk(sh->mu);
    if (!sh->cv.wait_until(lk, start_began + std::chrono::seconds(10), [&] { return sh->done; })) {
      if (!quiet()) std::fprintf(stderr, "host_gpu: the driver did not answer within 10 s (an address outside the card's range hangs on some systems); mode off\n");
      start_failed = dead = true;
      return false;
    }
    if (!sh->ok) {
      if (!quiet()) std::fprintf(stderr, "host_gpu: %s; mode off\n", sh->why.c_str());
      start_failed = dead = true;
      return false;
    }
    window_base = va_base;
    mapped_end = va_base;
    start_ready = true;
    return true;
  }

  // At exit: a driver start that is still running must not race the process's teardown (the real driver's own exit handlers). Bounded by the same deadline.
  void settle_start() {
    if (!start_sh || early_start == 2) return;
    std::unique_lock<std::mutex> lk(start_sh->mu);
    start_sh->cv.wait_until(lk, start_began + std::chrono::seconds(10), [&] { return start_sh->done; });
  }

  // The granules of [lo, hi) are backed by physical memory on the card (filled with zeros when first mapped).
  bool ensure_mapped(uint64_t hi) {
    hi = (hi + kGranule - 1) & ~(kGranule - 1);
    if (hi <= mapped_end) return true;
    const uint64_t lo = mapped_end, size = hi - lo;
    size_t freeb = 0, total = 0;
    api.cuMemGetInfo(&freeb, &total);
    if (size + (256ull << 20) > freeb) return false;
    CUmemAllocationProp prop = {};
    prop.type = 1;   // CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = 1;   // CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = 0;
    CUhandle h;
    CUresult r = api.cuMemCreate(&h, size, &prop, 0);
    if (!r) r = api.cuMemMap(lo, size, 0, h, 0);
    CUmemAccessDesc ad = {};
    ad.location = prop.location;
    ad.flags = 3;   // CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    if (!r) r = api.cuMemSetAccess(lo, size, &ad, 1);
    if (r) {
      if (logging()) std::fprintf(stderr, "host_gpu: mapping 0x%llx + %llu MiB failed (%d)\n", static_cast<unsigned long long>(lo), static_cast<unsigned long long>(size >> 20), r);
      return false;
    }
    api.cuMemsetD8(lo, 0, size);
    mapped_end = hi;
    return true;
  }

  // VGPU_HOST_GPU_DIV=1: kernels whose only outside-the-set instructions are the IEEE division idiom (host_gpu.hpp division_idiom_only) are in the set. The
  // card's exhaustive division check (tools/probes/hostgpu_sfu_exhaustive.cu div) ran clean on an A10G (Round sp44); it stays a switch, off by default, because
  // no other card was run.
  bool div_idiom = false;

  static std::vector<CodeInsn> idiom_summary(const sass::Code& code) {
    std::vector<CodeInsn> out;
    out.reserve(code.count);
    for (size_t i = 0; i < code.count; ++i) {
      const sass::Instr& ins = code.instr(i);
      CodeInsn c;
      const bool nan_const = ins.mnemonic == "MUFU" && !ins.src.empty() && ins.src[0].kind == sass::Kind::FImm && ins.src[0].fwidth == 32 &&
                             (ins.src[0].fbits & 0x7f800000u) == 0x7f800000u && (ins.src[0].fbits & 0x7fffffu) != 0;
      c.kind = classify_for_idiom(ins.mnemonic, ins.mods, nan_const);
      c.pc = ins.pc;
      const auto reg_of = [&](size_t k) { return ins.src.size() > k && ins.src[k].kind == sass::Kind::Reg && !ins.src[k].neg && !ins.src[k].abs ? static_cast<int>(ins.src[k].reg) : -1; };
      if (c.kind == CodeInsn::Kind::Call && !ins.src.empty() && ins.src[0].kind == sass::Kind::Label) c.target = ins.src[0].imm;
      if (c.kind == CodeInsn::Kind::MufuRcp) c.src0 = reg_of(0);
      if (c.kind == CodeInsn::Kind::Fchk) { c.src0 = reg_of(0); c.src1 = reg_of(1); }
      out.push_back(c);
    }
    return out;
  }

  struct Decision {
    bool ok = false;
    const char* why = "";
    Refused kind = Refused::No;
    RelaxedRefusal rr = RelaxedRefusal::None;
    uint64_t instrs = 0;
    bool runnable = false, back_edge = false;
  };

  Decision kernel_eligible(const ptx::EntryFn& fn, const exec::LaunchConfig& cfg, const std::vector<std::vector<uint8_t>>& args, MemoryManager& mem) {
    Decision d;
    const auto unsafe = [&](const char* why) {
      d.why = why;
      d.kind = Refused::Safety;
      return d;
    };
    if (!fn.sass || fn.sass->image.empty()) return unsafe("not a SASS module");
    if (cfg.cooperative || cfg.cluster != std::array<uint32_t, 3>{0, 0, 0} || fn.req_cluster != std::array<uint32_t, 3>{0, 0, 0}) return unsafe("cooperative or cluster launch");
    if (cfg.textures) return unsafe("texture objects");
    if (cfg.conditionals) return unsafe("inside a graph with conditionals");
    if (cfg.warp_size && cfg.warp_size != 32) return unsafe("warp size");
    for (const auto& a : args)
      for (size_t i = 0; i + 8 <= a.size(); i += 8) {
        uint64_t v;
        std::memcpy(&v, &a[i], 8);
        if (v && mem.is_host_mapped(v)) return unsafe("argument points into mapped host memory");
      }
    // Device memory that is not in the allocation table (VMM mappings, IPC, the kernel heap) is not on the card: a pointer to it, from an argument or from
    // memory, would read what the card never got. Not cached: a mapping can appear between two launches of one kernel.
    if (const char* blocker = mem.card_blocker()) return unsafe(blocker);
    const sass::Module& m = *fn.sass;
    const auto key = std::make_pair(&m, fn.name);
    const auto decision_of = [&](const Verdict& v) {
      d.ok = v.ok;
      d.why = v.why;
      d.kind = v.kind;
      d.rr = v.rr;
      d.instrs = v.instrs;
      d.runnable = v.runnable;
      d.back_edge = v.back_edge;
      return d;
    };
    if (const auto it = eligible.find(key); it != eligible.end()) return decision_of(it->second);
    const bool relaxed = mode == Mode::Relaxed;
    const bool say = logging() && !relaxed;   // exact mode's log lines, as they always were; relaxed mode says its own below
    Verdict v;
    bool ok = true;
    const char* why = "";
    std::vector<std::pair<uint64_t, uint64_t>> branches;
    if (!m.symbol_va.empty() || !m.cubin.externs.empty() || m.device_launches) { ok = false; why = "module variables, device calls or device-side launches"; }
    const auto ci = m.code_index.find(".text." + fn.name);
    if (ok && ci == m.code_index.end()) { ok = false; why = "no code section"; }
    const bool runnable_here = ok;
    if (!ok) {
      v.kind = Refused::Safety;
      v.why = why;   // (before the reason was kept, a later launch of such a kernel was said, and counted, as "not in the exact set")
    }
    ClassMask classes = 0;
    std::string first_never, first_gated;   // the text of the first instruction of no class, and of the first one a switch refuses
    if (ok) {
      const sass::Code& code = m.code[ci->second];
      v.instrs = code.count;
      // The classes of the code (vgpu/host_gpu.hpp relaxed_class) and its branches. Nothing is printed here and exact mode decides nothing by it.
      const uint64_t code_lo = code.count ? code.instr(0).pc : 0, code_hi = code.count ? code.instr(code.count - 1).pc : 0;
      for (size_t i = 0; i < code.count; ++i) {
        const sass::Instr& ins = code.instr(i);
        if (ins.op == sass::Op::BRA)
          for (const auto& o : ins.src)
            if (o.kind == sass::Kind::Label) branches.emplace_back(ins.pc, static_cast<uint64_t>(o.imm));
        InsnClass c = relaxed_class(ins.mnemonic, ins.mods);
        if (ins.op == sass::Op::CALL && c == InsnClass::Call) {
          // The scan has seen this section only: a call that leaves it (another section, a register, an absolute address) runs code nobody classed.
          const bool label = !ins.src.empty() && ins.src[0].kind == sass::Kind::Label;
          if (!call_stays_inside(mods_have(ins.mods, "REL"), label, label ? ins.src[0].imm : -1, code_lo, code_hi)) c = InsnClass::Never;
        }
        classes |= class_bit(c);
        if (c == InsnClass::Never && first_never.empty()) first_never = sass::to_text(ins);
        const ClassMask gated = sw.off | (sw.atomics ? 0 : class_bit(InsnClass::AtomFloat) | class_bit(InsnClass::AtomSwap));
        if ((class_bit(c) & gated) && first_gated.empty()) first_gated = sass::to_text(ins);
      }
      // The exact set's scan, unchanged.
      bool idiom_checked = false, idiom_ok = false;
      for (size_t i = 0; i < code.count && ok; ++i) {
        const sass::Instr& ins = code.instr(i);
        if (!exact_opcode_allowed(ins.mnemonic, ins.mods)) {
          if (div_idiom && (ins.op == sass::Op::MUFU || ins.op == sass::Op::CALL || ins.op == sass::Op::RET)) {
            if (!idiom_checked) {
              idiom_checked = true;
              const char* idiom_why = "";
              idiom_ok = division_idiom_only(idiom_summary(code), &idiom_why);
              if (!idiom_ok && say) {
                std::fprintf(stderr, "host_gpu: %s: not the division idiom: %s\n", fn.name.c_str(), idiom_why);
                for (const CodeInsn& c : idiom_summary(code))
                  if (c.kind != CodeInsn::Kind::Other) std::fprintf(stderr, "host_gpu:   pc 0x%llx kind %d src0 %d src1 %d target %lld\n", static_cast<unsigned long long>(c.pc), static_cast<int>(c.kind), c.src0, c.src1, static_cast<long long>(c.target));
              }
            }
            if (idiom_ok) continue;
          }
          ok = false;
          why = "instruction outside the exact set";
          if (say) std::fprintf(stderr, "host_gpu: %s refused: %s\n", fn.name.c_str(), sass::to_text(ins).c_str());
        }
      }
    }
    if (!ok && runnable_here && say) {   // what the refusal is made of: every MUFU form of the kernel, and whether it holds a call (the division and sqrt slow paths)
      std::map<std::string, unsigned> sfu;
      unsigned calls = 0, fchk = 0;
      const sass::Code& code = m.code[ci->second];
      for (size_t i = 0; i < code.count; ++i) {
        const sass::Instr& ins = code.instr(i);
        if (ins.mnemonic == "MUFU") {
          std::string k;
          for (const std::string& md : ins.mods) k += (k.empty() ? "" : ".") + md;
          ++sfu[k.empty() ? "?" : k];
        }
        calls += ins.mnemonic == "CALL";
        fchk += ins.mnemonic == "FCHK";
      }
      std::string line;
      for (const auto& [k, n] : sfu) line += " MUFU." + k + "x" + std::to_string(n);
      std::fprintf(stderr, "host_gpu: %s: SFU forms:%s; %u FCHK, %u CALL\n", fn.name.c_str(), line.empty() ? " none" : line.c_str(), fchk, calls);
      if (logging_code() && !sfu.empty())
        for (size_t i = 0; i < code.count; ++i) std::fprintf(stderr, "host_gpu:   %s: %s\n", fn.name.c_str(), sass::to_text(code.instr(i)).c_str());
    }
    if (runnable_here) {
      const bool exact_ok = ok;
      v.rr = exact_ok ? RelaxedRefusal::None : relaxed_refusal(classes, sw);
      if (relaxed) ok = v.rr == RelaxedRefusal::None;   // (a kernel of the exact set, the division idiom included, is admitted whatever is switched off)
      if (!ok) {
        v.kind = Refused::Code;
        static const char* const kRelaxedWhy[] = {"", "an instruction of no relaxed class", "a class switched off with VGPU_HOST_GPU_RELAXED_OFF",
                                                  "an atomic that needs VGPU_HOST_GPU_ATOMICS=1"};
        why = relaxed ? kRelaxedWhy[static_cast<unsigned>(v.rr)] : why;
        v.why = relaxed ? why : "kernel code not in the exact set (cached)";
      }
      if (relaxed) {
        // One line per kernel, for the benchmark to list what relaxed mode still cannot send: said like the summary (not under VGPU_QUIET alone).
        static const char* const kRefusalName[] = {"", "never", "off", "atomics"};
        if (!ok && (!quiet() || logging()))
          std::fprintf(stderr, "host_gpu_refused: kernel=%s class=%s insn=%s\n", fn.name.c_str(), kRefusalName[static_cast<unsigned>(v.rr)],
                       (v.rr == RelaxedRefusal::Never ? first_never : first_gated).c_str());
        if (logging()) {
          std::string held;
          for (unsigned c = 0; c < kInsnClasses; ++c)
            if (classes & (ClassMask{1} << c)) held += std::string(held.empty() ? "" : " ") + insn_class_name(static_cast<InsnClass>(c));
          std::fprintf(stderr, "host_gpu: %s: classes: %s; %s\n", fn.name.c_str(), held.c_str(), ok ? (exact_ok ? "admitted (exact set)" : "admitted") : why);
        }
      }
    }
    v.ok = ok;
    v.runnable = runnable_here;
    v.back_edge = has_back_edge(branches);
    eligible[key] = v;
    decision_of(v);
    d.why = why;   // the first launch says the reason in full
    return d;
  }

  CUfunction function_for(const ptx::EntryFn& fn) {
    const sass::Module* m = fn.sass.get();
    CUmodule mod;
    if (const auto it = modules.find(m); it != modules.end()) mod = it->second;
    else {
      if (const CUresult r = api.cuModuleLoadData(&mod, m->image.data())) {
        if (logging()) std::fprintf(stderr, "host_gpu: cuModuleLoadData failed (%d)\n", r);
        return nullptr;
      }
      modules[m] = mod;
    }
    CUfunction f = nullptr;
    if (api.cuModuleGetFunction(&f, mod, fn.name.c_str())) return nullptr;
    return f;
  }
};

Runtime::Runtime() : impl_(new Impl) {}
Runtime::~Runtime() { delete impl_; }

namespace {
void print_summary(Runtime::Impl* im) {
  if (!im || im->start_failed) return;   // (a driver that could not be used already said "mode off")
  // VGPU_QUIET silences the prose, not a report that was asked for: VGPU_HOST_GPU_VERIFY and VGPU_HOST_GPU_LOG each exist to be read at the end of the run.
  if (quiet() && !im->verify && !logging()) return;
  const Stats& s = im->stats;
  const bool relaxed = im->mode == Mode::Relaxed;
  const auto ull = [](uint64_t v) { return static_cast<unsigned long long>(v); };
  if (relaxed)
    std::fprintf(stderr,
                 "host_gpu: %llu launches ran on the card, %llu in the simulator (%llu kernels not admitted by relaxed mode, %llu too small, %llu unsupported launch shapes, %llu failed on the card)\n",
                 ull(s.offloaded), ull(s.simulated), ull(s.refused_kernel), ull(s.refused_size), ull(s.refused_launch), ull(s.failed));
  else
    std::fprintf(stderr,
                 "host_gpu: %llu launches ran on the card, %llu in the simulator (%llu kernels not in the exact set, %llu too small, %llu unsupported launch shapes, %llu failed on the card)\n",
                 ull(s.offloaded), ull(s.simulated), ull(s.refused_kernel), ull(s.refused_size), ull(s.refused_launch), ull(s.failed));
  std::fprintf(stderr, "host_gpu: %.1f MiB went to the card, %.1f MiB came back\n", s.pushed_bytes / 1048576.0, s.pulled_bytes / 1048576.0);
  if (im->verify)   // (the second line is for scripts: the card CI parses it, not the prose above)
    std::fprintf(stderr, "host_gpu_verify: in_set_identical=%llu in_set_different=%llu other_identical=%llu other_different=%llu\n", ull(s.same_exact_set),
                 ull(s.differ_exact_set), ull(s.same_other), ull(s.differ_other));
  if (im->verify && !relaxed)
    std::fprintf(stderr, "host_gpu: verify: kernels in the exact set: %llu identical, %llu different; other kernels: %llu identical, %llu different\n",
                 ull(s.same_exact_set), ull(s.differ_exact_set), ull(s.same_other), ull(s.differ_other));
  // The two lines for scripts (tools/bench/hostgpu/run.py): always in relaxed mode; in exact mode only with VGPU_HOST_GPU_LOG, so that exact mode's
  // summary is what it was. launches_card counts every launch that ran on the card, the ones a verify run repeats in the simulator included.
  if (relaxed || logging()) {
    uint64_t kernels_outside = 0;
    for (const auto& [k, kt] : im->tol_by_kernel) kernels_outside += kt.rep.outside > 0;
    std::fprintf(stderr,
                 "host_gpu_relaxed: mode=%s launches_card=%llu launches_sim=%llu card_s=%.6f sim_s=%.6f words_compared=%llu words_different=%llu words_outside=%llu words_subnormal=%llu kernels_outside=%llu max_abs=%.9g max_rel=%.9g\n",
                 relaxed ? "relaxed" : "exact", ull(s.card_runs), ull(s.simulated), s.card_seconds, s.sim_seconds, ull(s.tol.words), ull(s.tol.different), ull(s.tol.outside),
                 ull(s.tol.subnormal), ull(kernels_outside), s.tol.max_abs, s.tol.max_rel);
    std::fprintf(stderr, "host_gpu_refusals: refused_never=%llu refused_atomics=%llu refused_off=%llu refused_exact=%llu refused_safety=%llu too_small=%llu failed=%llu\n",
                 ull(s.refused_never), ull(s.refused_atomics), ull(s.refused_off), ull(s.refused_exact), ull(s.refused_launch), ull(s.refused_size), ull(s.failed));
    std::fprintf(stderr, "host_gpu_sampled: sampled=%llu sent=%llu timeouts=%llu cost_s=%.6f\n", ull(s.sampled), ull(s.sample_sent), ull(s.sample_timeouts), s.sample_seconds);
    if (relaxed && im->verify)
      for (const auto& [k, kt] : im->tol_by_kernel)
        if (kt.rep.outside)
          std::fprintf(stderr, "host_gpu_outside: kernel=%s words_compared=%llu words_different=%llu words_outside=%llu max_abs=%.9g max_rel=%.9g first=0x%llx:card=0x%08x:sim=0x%08x\n",
                       k.c_str(), ull(kt.rep.words), ull(kt.rep.different), ull(kt.rep.outside), kt.rep.max_abs, kt.rep.max_rel, ull(kt.rep.first_addr), kt.rep.first_card,
                       kt.rep.first_sim);
  }
}
Runtime::Impl* g_summary_impl = nullptr;
}  // namespace

Runtime* Runtime::for_device(const DeviceProfile& profile, int ordinal) {
  static std::once_flag once;
  static Runtime* rt = nullptr;
  std::call_once(once, [&] {
    const char* mode_text = std::getenv("VGPU_HOST_GPU");
    if (!mode_text_is_known(mode_text)) std::fprintf(stderr, "host_gpu: VGPU_HOST_GPU=%s is not a mode (off, exact or relaxed); mode off\n", mode_text);
    const Mode mode = mode_from_text(mode_text);
    if (mode == Mode::Off) return;
    // Said even under VGPU_QUIET: someone who asked for the mode and got none must be told why.
    for (const char* v : kIncompatibleVars)
      if (diagnostic_switched_on(v, std::getenv(v))) {
        std::fprintf(stderr, "host_gpu: %s is set and needs the simulator's launches; mode off\n", v);
        return;
      }
    if (profile.vendor != "nvidia") {
      if (!quiet()) std::fprintf(stderr, "host_gpu: only NVIDIA devices; mode off\n");
      return;
    }
    // Relaxed mode's own switches. A value that cannot be read turns the mode off with a message (said even under VGPU_QUIET): someone who asked for a
    // tolerance or for a class to be off must not get a run under another.
    RelaxedSwitches sw;
    Tolerance tol;
    double timeout_s = mode == Mode::Relaxed ? 600 : 0;
    if (mode == Mode::Relaxed) {
      if (const char* v = std::getenv("VGPU_HOST_GPU_ATOMICS"); v && *v && *v != '0') sw.atomics = true;
      std::string bad;
      if (!parse_relaxed_off(std::getenv("VGPU_HOST_GPU_RELAXED_OFF"), &sw.off, &bad)) {
        std::fprintf(stderr, "host_gpu: VGPU_HOST_GPU_RELAXED_OFF: %s is not a class (mufu, tensor, half, call, atomics, red, extra); mode off\n", bad.c_str());
        return;
      }
      const std::pair<const char*, double*> tolerances[] = {{"VGPU_HOST_GPU_ATOL", &tol.atol}, {"VGPU_HOST_GPU_RTOL", &tol.rtol}};
      for (const auto& [name, to] : tolerances)
        if (const char* v = std::getenv(name); v && *v && !parse_tolerance(v, to)) {
          std::fprintf(stderr, "host_gpu: %s=%s is not a number that is zero or more; mode off\n", name, v);
          return;
        }
      if (const char* v = std::getenv("VGPU_HOST_GPU_TOL_TYPE"); v && *v && !parse_reading(v, &tol.reading)) {
        std::fprintf(stderr, "host_gpu: VGPU_HOST_GPU_TOL_TYPE=%s is not auto, f32, f64 or f16; mode off\n", v);
        return;
      }
    }
    if (const char* v = std::getenv("VGPU_HOST_GPU_TIMEOUT"); v && *v && !parse_tolerance(v, &timeout_s)) {   // (the same form: seconds, zero or more; 0 is no limit)
      std::fprintf(stderr, "host_gpu: VGPU_HOST_GPU_TIMEOUT=%s is not a number of seconds that is zero or more; mode off\n", v);
      return;
    }
    auto* r = new Runtime;
    r->impl_->mode = mode;
    r->impl_->sw = sw;
    r->impl_->tol = tol;
    r->impl_->watchdog.limit_s = timeout_s;
    // The driver is not started here: see Impl::begin_start. The switches are read now so that they do not depend on whether it came up.
    r->impl_->configure_start(profile, kDeviceVaBase);
    if (const char* v = std::getenv("VGPU_HOST_GPU_VERIFY"); v && *v && *v != '0') r->impl_->verify = std::string(v) == "all" ? 2 : 1;
    if (const char* v = std::getenv("VGPU_HOST_GPU_WRITE_THROUGH"); v && *v && *v != '0') r->impl_->resident = false;
    if (const char* v = std::getenv("VGPU_HOST_GPU_NO_FIRST_LAUNCH"); v && *v && *v != '0') r->impl_->first_launch_off = true;
    if (const char* v = std::getenv("VGPU_HOST_GPU_DIV"); v && *v && *v != '0') r->impl_->div_idiom = true;
    if (const char* v = std::getenv("VGPU_HOST_GPU_EARLY_START"); v && *v && *v != '0') r->impl_->early_start = *v == '2' ? 2 : 1;
    if (const char* v = std::getenv("VGPU_HOST_GPU_SAMPLE"); v && *v)   // 0: no launch is sampled (the rule before sp60b); log: see SampleMode
      r->impl_->sample_mode = std::string(v) == "0" ? Impl::SampleMode::Off : std::string(v) == "log" ? Impl::SampleMode::Log :
                              std::string(v) == "always" ? Impl::SampleMode::Always : Impl::SampleMode::On;
    if (const char* v = std::getenv("VGPU_HOST_GPU_STATIC"); v && *v && *v != '0') r->impl_->static_estimate = true;
    if (const char* v = std::getenv("VGPU_HOST_GPU_START_COST"); v && *v) {   // seconds; 0: do not charge the driver start (the rule before sp43)
      char* end = nullptr;
      const double sc = std::strtod(v, &end);
      if (end != v && sc >= 0) r->impl_->costs.start_cost_s = sc;
    }
    if (const char* mb = std::getenv("VGPU_HOST_GPU_MAX_MIB")) r->impl_->max_bytes = std::strtoull(mb, nullptr, 10) << 20;
    g_summary_impl = r->impl_;
    std::atexit([] {
      g_summary_impl->settle_start();
      print_summary(g_summary_impl);
    });
    rt = r;
  });
  return ordinal == 0 ? rt : nullptr;
}

void Runtime::before_simulated(MemoryManager& mem) {
  if (mem.card_resident_count().load(std::memory_order_relaxed) <= 0) {
    impl_->verify_card_ran = false;
    return;
  }
  std::lock_guard<GateLock> g(impl_->mu);
  if (impl_->verify_card_ran) {
    // Verifying: this launch already ran on the card and the simulator runs it now, on its own memory, which is the truth. The card's copy is given up
    // (nothing is pulled). Until Round sp59 it was kept valid, which was wrong twice over: the simulator then ran its launch through the gate's slow
    // path, where an atomic of one host thread could lose the update of another (HeCBench atomicAggregate printed FAIL under VERIFY with 4 host threads
    // and PASS with 1), and the next launch on the card started from the card's own earlier results instead of the simulator's, so in relaxed mode the
    // differences of one launch carried into the next. The price is that a verify run pushes all live memory before every launch.
    impl_->verify_card_ran = false;
    for (auto& [b, a] : impl_->res) {
      if (a.slot->load(std::memory_order_relaxed) == 0) continue;
      a.r = Residency(a.r.chunks());
      impl_->update_slot(a);
    }
    return;
  }
  impl_->reset_for_simulator();
}

void Runtime::note_simulated(const ptx::EntryFn& fn, const exec::LaunchConfig& cfg, double sim_seconds, const MemoryManager* mem,
                             const std::vector<std::vector<uint8_t>>* args) {
  std::lock_guard<GateLock> g(impl_->mu);
  impl_->stats.sim_seconds += sim_seconds;
  if (impl_->start_failed) return;   // the driver could not be used: nothing to learn for
  InternalScope internal;
  if (g_trace_file) {
    std::string bases;
    if (mem && args)
      for (const auto& a : *args)
        for (size_t i = 0; i + 8 <= a.size(); i += 8) {
          uint64_t v, b, sz;
          std::memcpy(&v, &a[i], 8);
          if (v && mem->find_allocation(v, &b, &sz)) {
            char t[24];
            std::snprintf(t, sizeof t, " %llx", static_cast<unsigned long long>(b));
            bases += t;
          }
        }
    std::lock_guard<std::mutex> tg(g_trace_mu);
    std::fprintf(g_trace_file, "L %s %llu %.9f %d %.9f |%s\n", fn.name.c_str(), static_cast<unsigned long long>(uint64_t{cfg.grid[0]} * cfg.grid[1] * cfg.grid[2]),
                 sim_seconds, impl_->last.in_set ? 1 : 0, impl_->last.card_s, bases.c_str());
  }
  Impl::Pending& p = impl_->pending;
  if (p.active && mem && p.kernel == fn.name) {
    const bool relaxed = impl_->mode == Mode::Relaxed;
    bool same = true;
    uint64_t bad_chunks = 0;
    TolReport rep;   // relaxed mode: the differing chunks word by word (vgpu/host_gpu_tol.hpp); equal chunks only add to the count of words
    std::vector<uint8_t> sim_bytes;
    for (size_t i = 0; i < p.ranges.size(); ++i) {
      sim_bytes.resize(p.ranges[i].second);
      mem->read(p.ranges[i].first, sim_bytes.data(), p.ranges[i].second);
      for (uint64_t c = 0; c < sim_bytes.size(); c += kChunkSize) {
        const uint64_t n = std::min<uint64_t>(kChunkSize, sim_bytes.size() - c);
        if (std::memcmp(sim_bytes.data() + c, p.card[i].data() + c, n) != 0) {
          same = false;
          ++bad_chunks;
          if (relaxed) compare_words(p.card[i].data() + c, sim_bytes.data() + c, p.ranges[i].first + c, n, impl_->tol, rep);
        } else if (relaxed) {
          rep.words += tol_word_count(p.ranges[i].first + c, n);
        }
      }
    }
    Stats& st = impl_->stats;
    ++(p.in_exact_set ? (same ? st.same_exact_set : st.differ_exact_set) : (same ? st.same_other : st.differ_other));
    if (relaxed) {
      st.tol.add(rep);
      Impl::KernelTol& kt = impl_->tol_by_kernel[fn.name];
      kt.rep.add(rep);
      // Every launch with VGPU_HOST_GPU_LOG; without it, the first three launches of a kernel that have a word outside tolerance.
      constexpr unsigned kSaidWithoutLog = 3;
      const bool say_outside = rep.outside && kt.said < kSaidWithoutLog;
      if (say_outside) ++kt.said;
      if (logging() || say_outside) {
        char first[96] = "";
        if (rep.has_first)
          std::snprintf(first, sizeof first, ", first outside at 0x%llx: card 0x%08x sim 0x%08x", static_cast<unsigned long long>(rep.first_addr), rep.first_card, rep.first_sim);
        std::fprintf(stderr, "host_gpu: verify %s (%s): %llu words compared, %llu different, %llu outside tolerance, %llu zero or denormal on both sides; largest difference %.9g, relative %.9g%s%s\n",
                     fn.name.c_str(), p.in_exact_set ? "admitted" : "not admitted", static_cast<unsigned long long>(rep.words), static_cast<unsigned long long>(rep.different),
                     static_cast<unsigned long long>(rep.outside), static_cast<unsigned long long>(rep.subnormal), rep.max_abs, rep.max_rel, first,
                     say_outside && !logging() && kt.said == kSaidWithoutLog ? " (later launches of this kernel are not listed; VGPU_HOST_GPU_LOG=1 lists all)" : "");
      }
    } else if (logging() || (!same && p.in_exact_set)) {
      std::fprintf(stderr, "host_gpu: verify %s (%s): %s%s\n", fn.name.c_str(), p.in_exact_set ? "in the exact set" : "outside the exact set",
                   same ? "identical" : "DIFFERENT, 64 KiB chunks: ", same ? "" : std::to_string(bad_chunks).c_str());
    }
    p = {};
  }
  if (t_last_sample.set) {   // this launch was sampled and the simulator kept it: what the sample said against what the launch took
    const LastSample& l = t_last_sample;
    if (logging() || impl_->sample_mode == Impl::SampleMode::Log)
      std::fprintf(stderr,
                   "host_gpu_sample: kernel=%s blocks=%llu threads_per_block=%u instrs=%llu back_edge=%d whole_samples=%u live_bytes=%llu static_s=%.6f age_s=%.6f affordable=%d answered=%d ran=%llu first_block_s=%.6f timed_s=%.6f cost_s=%.6f fork_s=%.6f threads=%u estimate_s=%.6f would_send=%d launch_s=%.6f\n",
                   fn.name.c_str(), static_cast<unsigned long long>(l.hint.blocks), l.hint.threads_per_block, static_cast<unsigned long long>(l.hint.instrs),
                   l.hint.back_edge ? 1 : 0, l.hint.samples, static_cast<unsigned long long>(l.hint.live), l.hint.static_s, l.hint.age_s, l.hint.affordable ? 1 : 0, l.answered ? 1 : 0,
                   static_cast<unsigned long long>(l.ran), l.first_block_s, l.sample.seconds, l.wall_s, l.fork_s, l.sample.threads, l.estimate_s, l.would_send ? 1 : 0,
                   sim_seconds);
    t_last_sample = {};
  }
  impl_->sim_times.record(fn.name, uint64_t{cfg.grid[0]} * cfg.grid[1] * cfg.grid[2], sim_seconds);
  if (impl_->last.in_set) impl_->eligible_sim_spent_s += sim_seconds;   // an exact-set kernel that the card could have run
}

bool Runtime::try_launch(const ptx::EntryFn& fn, const exec::LaunchConfig& cfg, const std::vector<std::vector<uint8_t>>& args,
                         MemoryManager& mem) {
  Impl& im = *impl_;
  std::lock_guard<GateLock> g(im.mu);   // one offloaded launch at a time: the card path shares the reserved window
  t_sample_hint = {};
  t_last_sample = {};
  return im.launch(fn, cfg, args, mem, nullptr);
}

// The decision for a launch and, if it goes, the card path (mu held). `forced_sim_s`: a sampled launch that sample_decide sends to the card: nothing
// is decided, and the value is the simulator's time its sample gave.
bool Runtime::Impl::launch(const ptx::EntryFn& fn, const exec::LaunchConfig& cfg, const std::vector<std::vector<uint8_t>>& args, MemoryManager& mem,
                           const double* forced_sim_s) {
  Impl& im = *this;
  const bool forced = forced_sim_s != nullptr;
  InternalScope internal;
  // (a forced launch was counted as the simulator's when it was first decided; sample_decide takes that count back if the card runs it)
  struct KeepCount {
    uint64_t& v;
    const uint64_t was;
    const bool on;
    ~KeepCount() { if (on) v = was; }
  } keep_simulated{im.stats.simulated, im.stats.simulated, forced};
  im.last = {};
  if (im.dead) return false;
  im.pending = {};
  const Impl::Decision dec = im.kernel_eligible(fn, cfg, args, mem);
  const char* const why = dec.why;
  const uint64_t static_instrs = dec.instrs;
  const bool runnable = dec.runnable, back_edge = dec.back_edge;
  const bool in_exact_set = dec.ok;   // (in relaxed mode: admitted by its classes)
  im.last.in_set = in_exact_set;
  const uint64_t blocks = uint64_t{cfg.grid[0]} * cfg.grid[1] * cfg.grid[2];
  // (The driver is not started here, whatever the kernel: only a launch that has been decided to go to the card starts it, below, at await_start.)
  if (!in_exact_set && !(im.verify == 2 && runnable)) {
    if (dec.kind == Impl::Refused::Code) {
      ++im.stats.refused_kernel;
      ++(dec.rr == RelaxedRefusal::Never ? im.stats.refused_never : dec.rr == RelaxedRefusal::Off ? im.stats.refused_off : dec.rr == RelaxedRefusal::Atomics ? im.stats.refused_atomics : im.stats.refused_exact);
    } else {
      ++im.stats.refused_launch;
    }
    ++im.stats.simulated;
    if (logging()) std::fprintf(stderr, "host_gpu: %s stays in the simulator: %s\n", fn.name.c_str(), why);
    return false;
  }
  if (im.early_start && !forced && !im.verify && !im.start_sh && !im.start_failed && back_edge && im.multiprocessors &&
      blocks >= kFirstLaunchBlocksPerSm * im.multiprocessors)
    im.begin_start();
  // What the simulator took before for this kernel, else a lower bound from the code and the grid.
  std::optional<double> sim_s = forced ? std::optional<double>(*forced_sim_s) : im.sim_times.predict(fn.name, blocks);
  if (!sim_s && im.static_estimate) sim_s = static_sim_seconds(blocks, cfg.block[0] * cfg.block[1] * cfg.block[2], static_instrs, im.costs);
  if (!sim_s && im.verify) sim_s = 0;   // verifying: every launch runs on both, whatever the rule would say
  bool first_launch_rule = false;
  if (!sim_s && !im.verify && im.sim_times.samples(fn.name) == 0 && im.card_kernel_s.count(fn.name) == 0 && !im.first_launch_off) {
    // Nothing is known of this kernel: the conservative first-launch rule (host_gpu.hpp) may send this one launch to the card.
    uint64_t live = 0;
    for (const auto& r : mem.live_ranges()) live += r.second;
    const FirstLaunch fl{blocks, cfg.block[0] * cfg.block[1] * cfg.block[2], static_instrs, back_edge, im.multiprocessors, live};
    if (first_launch_offload(fl, im.costs, im.start_sh != nullptr, im.eligible_sim_spent_s)) {
      sim_s = static_sim_seconds(blocks, fl.threads_per_block, static_instrs, im.costs);
      first_launch_rule = true;
    }
  }
  if (!sim_s) {   // the first launches of a kernel are the simulator's: they teach the estimate (VGPU_HOST_GPU_STATIC=1 is the exception, and is not sound)
    im.consider_sample(fn, cfg, dec, blocks, mem);   // ... unless the time of its first blocks sends this one to the card (sample_plan, sample_decide)
    ++im.stats.simulated;
    return false;
  }
  const auto ranges = mem.live_ranges();
  uint64_t total = 0, hi = 0;
  for (const auto& r : ranges) {
    total += r.second;
    hi = std::max(hi, r.first + r.second);
  }
  uint64_t push_bytes = total;
  if (im.resident) {   // only the chunks the simulator wrote since the card last had them; what the host reads later is paid when it reads
    im.sync_tracking(mem, ranges);
    push_bytes = 0;
    for (const auto& [b, a] : im.res)
      for (uint64_t c : a.r.host_chunks()) push_bytes += Impl::chunk_len(a, c);
  }
  Estimate e;
  e.sim_s = *sim_s;
  e.push_bytes = push_bytes;
  e.pull_bytes = im.resident ? 0 : total;   // write-through: an upper bound, only what changed is written back, but the card's copy is read in full
  if (const auto it = im.card_kernel_s.find(fn.name); it != im.card_kernel_s.end()) e.card_kernel_s = it->second;
  if (!forced && !im.verify && im.banned.count(fn.name)) {
    ++im.stats.refused_size;
    ++im.stats.simulated;
    return false;
  }
  if (total > im.max_bytes || (!forced && !im.verify && !should_offload_with_start(e, im.costs, im.start_sh != nullptr, im.eligible_sim_spent_s))) {
    ++im.stats.refused_size;
    ++im.stats.simulated;
    if (logging()) std::fprintf(stderr, "host_gpu: %s stays in the simulator (sim %.3f ms, card path %.3f ms%s, %.1f MiB)\n", fn.name.c_str(), e.sim_s * 1e3, card_path_seconds(e, im.costs) * 1e3, im.start_sh ? "" : " without the driver start", total / 1048576.0);
    return false;
  }

  // The card path. Anything that goes wrong before the kernel has run leaves the simulator's memory as it was.
  if (!im.await_start()) {   // the driver is unusable (said once on stderr): this launch and all later ones are the simulator's
    ++im.stats.simulated;
    return false;
  }
  const auto t0 = std::chrono::steady_clock::now();
  im.api.cuCtxSetCurrent(im.ctx);
  const uint64_t base = mem.va_base();
  if (hi < base || !im.ensure_mapped(hi)) {
    if (logging()) std::fprintf(stderr, "host_gpu: %s: could not back the simulator's window up to 0x%llx on the card\n", fn.name.c_str(), static_cast<unsigned long long>(hi));
    ++im.stats.failed;
    ++im.stats.simulated;
    return false;
  }
  CUfunction f = im.function_for(fn);
  if (!f) {
    if (logging()) std::fprintf(stderr, "host_gpu: %s: the real driver did not load the module or find the kernel\n", fn.name.c_str());
    ++im.stats.failed;
    ++im.stats.simulated;
    return false;
  }
  std::vector<uint8_t>& buf = im.push_buf;
  bool card_only_data = false;   // chunks only the card holds, from earlier launches: lost if the card fails now
  if (im.resident) {
    for (auto& [b, a] : im.res) {
      card_only_data = card_only_data || a.r.count(Where::Card) > 0;
      for (const auto& [first, count] : chunk_runs(a.r.host_chunks()))
        for (uint64_t c = first; c < first + count;) {
          const uint64_t n_chunks = std::min<uint64_t>(first + count - c, (32ull << 20) / kChunkSize);
          const uint64_t off = c * kChunkSize, bytes = std::min<uint64_t>(n_chunks * kChunkSize, a.size - off);
          buf.resize(bytes);
          mem.read(a.base + off, buf.data(), bytes);
          if (im.api.cuMemcpyHtoD(a.base + off, buf.data(), bytes)) {
            im.dead = true;
            ++im.stats.failed;
            ++im.stats.simulated;
            if (card_only_data) throw Error(Err::Internal, "host_gpu: copying to the card failed while it holds data the simulator does not");
            im.reset_for_simulator();
            return false;
          }
          im.stats.pushed_bytes += bytes;
          c += n_chunks;
        }
      a.r.card_prepare();   // the pushed chunks are Both now
    }
  } else {
    for (const auto& [rb, rs] : ranges)
      for (uint64_t at = 0; at < rs;) {
        const uint64_t n = std::min<uint64_t>(rs - at, 32ull << 20);
        buf.resize(n);
        mem.read(rb + at, buf.data(), n);
        if (im.api.cuMemcpyHtoD(rb + at, buf.data(), n)) { im.dead = true; ++im.stats.failed; ++im.stats.simulated; return false; }
        im.stats.pushed_bytes += n;
        at += n;
      }
  }
  if (cfg.shared_bytes > 48 * 1024) im.api.cuFuncSetAttribute(f, kFuncAttrMaxDynamicShared, static_cast<int>(cfg.shared_bytes));
  std::vector<void*> params;
  std::vector<std::vector<uint8_t>> keep = args;
  for (auto& a : keep) params.push_back(a.data());
  const auto k0 = std::chrono::steady_clock::now();
  im.watchdog.arm(fn.name);
  CUresult r = im.api.cuLaunchKernel(f, cfg.grid[0], cfg.grid[1], cfg.grid[2], cfg.block[0], cfg.block[1], cfg.block[2], cfg.shared_bytes, nullptr,
                                     params.data(), nullptr);
  if (!r) r = im.api.cuCtxSynchronize();
  im.watchdog.disarm();
  const double kernel_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - k0).count();
  if (r) {   // a failed kernel poisons the context (an illegal address): the mode ends, the simulator takes this launch
    im.dead = true;
    ++im.stats.failed;
    ++im.stats.simulated;
    if (im.resident) {
      if (card_only_data) throw Error(Err::Internal, "host_gpu: " + fn.name + " failed on the card, which holds data from earlier launches that the simulator does not");
      im.reset_for_simulator();
    }
    if (!quiet()) std::fprintf(stderr, "host_gpu: %s failed on the card (error %d); mode off, the simulator runs it\n", fn.name.c_str(), r);
    return false;
  }
  im.last.card_s = kernel_s;
  ++im.stats.card_runs;
  // (the seconds of the card path, for the machine-readable summary: whatever way this function is left from here on)
  struct CardTime {
    Stats& st;
    std::chrono::steady_clock::time_point from;
    ~CardTime() { st.card_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - from).count(); }
  } card_time{im.stats, t0};
  if (im.resident) {   // the kernel may have written any live chunk: every one is the card's now
    std::vector<uint64_t> newly;
    for (auto& [b, a] : im.res) {
      const bool was = a.slot->load(std::memory_order_relaxed) > 0;
      a.r.card_launched();
      im.update_slot(a);
      if (!was) newly.push_back(a.base);
    }
    mem.card_void_lookups(newly);
  }
  if (im.verify) {   // keep the card's bytes; the simulator runs the launch next and note_simulated compares
    im.pending = {};
    im.pending.active = true;
    im.pending.in_exact_set = in_exact_set;
    im.pending.kernel = fn.name;
    im.pending.ranges = ranges;
    for (const auto& [rb, rs] : ranges) {
      im.pending.card.emplace_back(rs);
      for (uint64_t at = 0; at < rs; at += (32ull << 20))
        if (im.api.cuMemcpyDtoH(im.pending.card.back().data() + at, rb + at, std::min<uint64_t>(rs - at, 32ull << 20))) { im.pending = {}; return false; }
    }
    im.verify_card_ran = true;
    ++im.stats.simulated;
    return false;
  }
  if (im.resident) {   // nothing comes back now: the simulator's memory is stale where the card wrote, and the gate pulls it when it is read
    const double total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const bool first_card_launch = im.card_kernel_s.count(fn.name) == 0;
    if (!first_card_launch && !offload_paid_off(total_s, e.sim_s)) {
      im.banned.insert(fn.name);
      if (logging()) std::fprintf(stderr, "host_gpu: %s took %.3f ms on the card path against an estimate of %.3f ms in the simulator: offloading it ends\n", fn.name.c_str(), total_s * 1e3, e.sim_s * 1e3);
    }
    if (im.traffic.record(fn.name, push_bytes, total, total_s, e.sim_s, !first_card_launch) && im.banned.insert(fn.name).second && logging())
      std::fprintf(stderr, "host_gpu: %s pushed %.1f MiB of %.1f MiB live on each of its last %u launches: offloading it ends (traffic)\n", fn.name.c_str(),
                   push_bytes / 1048576.0, total / 1048576.0, kTrafficLaunches);
    im.card_kernel_s[fn.name] = kernel_s;
    ++im.stats.offloaded;
    im.sim_times.note_offload(fn.name);
    if (first_launch_rule && logging()) std::fprintf(stderr, "host_gpu: %s: first launch decided by the static rule (loop, %llu blocks)\n", fn.name.c_str(), static_cast<unsigned long long>(blocks));
    if (forced && logging()) std::fprintf(stderr, "host_gpu: %s: sent by the time of its first blocks (%.3f s in the simulator, at least)\n", fn.name.c_str(), *forced_sim_s);
    if (logging())
      std::fprintf(stderr, "host_gpu: %s on the card (resident): kernel %.3f ms, total %.3f ms (sim estimate %.3f ms), %.2f MiB pushed\n", fn.name.c_str(), kernel_s * 1e3,
                   total_s * 1e3, e.sim_s * 1e3, push_bytes / 1048576.0);
    return true;
  }
  // Back into the simulator: only the 64 KiB chunks whose bytes differ.
  std::vector<uint8_t>& sim_bytes = im.sim_buf;
  for (const auto& [rb, rs] : ranges)
    for (uint64_t at = 0; at < rs;) {
      const uint64_t n = std::min<uint64_t>(rs - at, 32ull << 20);
      buf.resize(n);
      sim_bytes.resize(n);
      if (im.api.cuMemcpyDtoH(buf.data(), rb + at, n)) {
        // The kernel has run: the simulator can no longer run it again on its own memory, which is half written back.
        im.dead = true;
        throw Error(Err::Internal, "host_gpu: reading the card's memory back failed after the kernel ran");
      }
      im.stats.pulled_bytes += n;
      mem.read(rb + at, sim_bytes.data(), n);
      for (uint64_t c = 0; c < n; c += kChunkSize) {
        const uint64_t m = std::min<uint64_t>(kChunkSize, n - c);
        if (std::memcmp(buf.data() + c, sim_bytes.data() + c, m) != 0) mem.write(rb + at + c, buf.data() + c, m);
      }
      at += n;
    }
  const double total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  // The first launch of a kernel on the card also pays the module's load: not held against the estimate.
  if (im.card_kernel_s.count(fn.name) && !offload_paid_off(total_s, e.sim_s)) {
    im.banned.insert(fn.name);
    if (logging()) std::fprintf(stderr, "host_gpu: %s took %.3f ms on the card path against an estimate of %.3f ms in the simulator: offloading it ends\n", fn.name.c_str(), total_s * 1e3, e.sim_s * 1e3);
  }
  im.card_kernel_s[fn.name] = kernel_s;
  ++im.stats.offloaded;
  im.sim_times.note_offload(fn.name);
  if (logging())
    std::fprintf(stderr, "host_gpu: %s on the card: kernel %.3f ms, total %.3f ms (sim estimate %.3f ms), %.1f MiB each way\n", fn.name.c_str(), kernel_s * 1e3,
                 total_s * 1e3, e.sim_s * 1e3, total / 1048576.0);
  return true;
}


// ---- a sampled launch (vgpu/host_gpu.hpp "The first blocks of a launch, timed") ---------------------------------------------------------

// Called by launch() for a launch that goes to the simulator because nothing is known of its kernel (mu held): leaves a hint for sample_plan.
// Not once the driver has been started: the process is forked for a sample, and a process that holds the real driver's context is not forked
// (the first-launch and history rules, which then find the start paid, are what is left).
void Runtime::Impl::consider_sample(const ptx::EntryFn& fn, const exec::LaunchConfig& cfg, const Decision& dec, uint64_t blocks, MemoryManager& mem) {
  if (sample_mode == SampleMode::Off || verify || !dec.ok || start_sh || start_failed || banned.count(fn.name)) return;
  uint64_t live = 0;
  for (const auto& r : mem.live_ranges()) live += r.second;
  if (live > max_bytes) return;
  Estimate e;
  e.push_bytes = e.pull_bytes = live;
  e.start_s = std::max(costs.start_cost_s, 0.0);
  SampleCandidate c;
  c.blocks = blocks;
  c.live_bytes = live;
  c.whole_s = sim_times.peek(fn.name, blocks);
  c.age_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_loaded).count();
  c.spent_s = sample_spent_s;
  c.expected_s = std::max(sample_dearest_s, kSampleCostGuessSeconds + kSampleForkSecondsPerByte * static_cast<double>(live));
  c.static_s = static_sim_seconds(blocks, cfg.block[0] * cfg.block[1] * cfg.block[2], dec.instrs, costs);
  const bool affordable = c.static_s >= sample_static_floor(live) || sample_affordable(c.age_s, c.spent_s, c.expected_s);
  if (sample_mode != SampleMode::On) c.age_s = 1e18;   // (log, always: neither the floor nor the process's age is asked)
  if (!worth_sampling(c, costs, card_path_seconds(e, costs))) return;
  SampleHint h;
  h.affordable = affordable;
  h.set = true;
  h.blocks = blocks;
  h.live = live;
  h.instrs = dec.instrs;
  h.threads_per_block = cfg.block[0] * cfg.block[1] * cfg.block[2];
  h.back_edge = dec.back_edge;
  h.static_s = c.static_s;
  h.samples = sim_times.samples(fn.name);
  h.age_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_loaded).count();
  t_sample_hint = h;
}

bool Runtime::sample_plan(const ptx::EntryFn&, SamplePlan* plan) {
  if (!t_sample_hint.set) return false;
  plan->limit = sample_blocks(t_sample_hint.blocks);
  plan->stride = sample_stride(t_sample_hint.blocks);
  plan->budget_s = kSampleBudgetSeconds;
  plan->timeout_s = kSampleTimeoutSeconds;
  return plan->limit != 0;
}

bool Runtime::sample_decide(const ptx::EntryFn& fn, const exec::LaunchConfig& cfg, const std::vector<std::vector<uint8_t>>& args, MemoryManager& mem,
                            const SampleRun& run) {
  Impl& im = *impl_;
  const SampleHint hint = t_sample_hint;
  t_sample_hint = {};
  if (!hint.set) return false;
  std::lock_guard<GateLock> g(im.mu);
  ++im.stats.sampled;
  im.stats.sample_seconds += run.wall_s;
  im.sample_spent_s += run.wall_s;
  im.sample_dearest_s = std::max(im.sample_dearest_s, run.wall_s);
  if (!run.answered) ++im.stats.sample_timeouts;
  // The measure is of the blocks after the first (which pays decode and set-up).
  Sample smp;
  smp.timed = run.answered && run.part.next > 1 && run.part.first_block_s >= 0 ? run.part.next - 1 : 0;
  smp.total = hint.blocks;
  smp.threads = std::max(1u, run.part.threads);
  smp.seconds = smp.timed ? std::max(run.part.run_s - run.part.first_block_s, 1e-9) : 0;
  const auto whole = sampled_launch_seconds(smp);
  // What the card would have to be sent: the chunks it does not hold (all of the live memory before its first launch).
  uint64_t live = 0, push = 0;
  const auto ranges = mem.live_ranges();
  for (const auto& r : ranges) live += r.second;
  push = live;
  if (im.resident && im.start_ready) {
    im.sync_tracking(mem, ranges);
    push = 0;
    for (const auto& [b, a] : im.res)
      for (uint64_t c : a.r.host_chunks()) push += Impl::chunk_len(a, c);
  }
  std::optional<double> card_kernel;
  if (const auto it = im.card_kernel_s.find(fn.name); it != im.card_kernel_s.end()) card_kernel = it->second;
  const bool want = whole && sample_offload(smp, push, live, im.costs, im.start_sh != nullptr, card_kernel);
  LastSample& l = t_last_sample;
  l = {};
  l.set = true;
  l.hint = hint;
  l.sample = smp;
  l.ran = run.part.next;   // (a copy that did not answer: how it ended, see sample_in_copy)
  l.first_block_s = run.part.first_block_s;
  l.wall_s = run.wall_s;
  l.fork_s = run.fork_s;
  l.estimate_s = whole.value_or(0);
  l.answered = run.answered;
  l.would_send = want;
  if (im.sample_mode == Impl::SampleMode::Log || !want || im.dead) return false;
  const double sim_s = *whole;
  if (!im.launch(fn, cfg, args, mem, &sim_s)) return false;   // the card could not (said by launch()): the simulator runs the launch
  ++im.stats.sample_sent;
  if (im.stats.simulated) --im.stats.simulated;   // (counted when the launch was first left to the simulator)
  im.sim_times.record_extrapolated(fn.name, hint.blocks, sim_s);
  if (logging())
    std::fprintf(stderr,
                 "host_gpu_sample: kernel=%s blocks=%llu threads_per_block=%u instrs=%llu back_edge=%d whole_samples=%u live_bytes=%llu static_s=%.6f age_s=%.6f affordable=%d answered=1 ran=%llu first_block_s=%.6f timed_s=%.6f cost_s=%.6f fork_s=%.6f threads=%u estimate_s=%.6f would_send=1 launch_s=-1\n",
                 fn.name.c_str(), static_cast<unsigned long long>(hint.blocks), hint.threads_per_block, static_cast<unsigned long long>(hint.instrs), hint.back_edge ? 1 : 0,
                 hint.samples, static_cast<unsigned long long>(hint.live), hint.static_s, hint.age_s, hint.affordable ? 1 : 0, static_cast<unsigned long long>(run.part.next),
                 run.part.first_block_s, smp.seconds, run.wall_s, run.fork_s, smp.threads, sim_s);
  t_last_sample = {};
  return true;
}

}  // namespace vgpu::hostgpu
