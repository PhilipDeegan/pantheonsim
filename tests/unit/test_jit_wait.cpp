// Tier 2, round sp57: the wait on the plain loop (src/sass/jit.hpp, State::Waiting). A section that has asked the cache and got nothing, and whose compile
// threshold is far, leaves the hook loop: the plain loop runs it and Unit::wait_note adds up what each call of it ran; `VGPU_JIT_PROFILE` instructions short
// of the threshold the section is Counting again, and the regions' counts of that last stretch (and of the stretch before the lookup) choose what is
// compiled. These tests drive the states without an executor (Unit::note and Unit::wait_note, as the run loop calls them).
//
// The executor side (words and counters across the four stretches against the interpreter) is ctest test_sass_chain_cont_jit_wait.
// Mutants (-DVGPU_JIT_MUTANTS=ON, VGPU_JIT_MUTANT=wait_never | wait_stuck | wait_no_credit | wait_no_profile): each makes the first test fail.
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "sass/jit.hpp"
#include "sass/tier.hpp"
#include "vgpu/sass/sass.hpp"
#include "vtest.hpp"

namespace fs = std::filesystem;
using namespace vgpu::sass;

namespace {

constexpr uint64_t kThreshold = 100000, kLookup = 128, kProfile = 10000;

struct EnvInit {
  fs::path dir;
  EnvInit() {
    const char* t = std::getenv("TMPDIR");
    dir = fs::path(t && *t ? t : "/tmp") / ("vgpu-test-jit-wait." + std::to_string(::getpid()));
    fs::create_directories(dir);
    ::setenv("VGPU_JIT", "1", 1);
    ::setenv("VGPU_JIT_SYNC", "1", 1);
    ::setenv("VGPU_JIT_THRESHOLD", std::to_string(kThreshold).c_str(), 1);
    ::setenv("VGPU_JIT_LOOKUP_AFTER", std::to_string(kLookup).c_str(), 1);
    ::setenv("VGPU_JIT_PROFILE", std::to_string(kProfile).c_str(), 1);
    ::setenv("VGPU_JIT_HOT_PERMILLE", "0", 1);   // (every region is compiled, and the share rule is off: the lookup is the first decision)
    ::setenv("VGPU_JIT_MIN_REGION", "2", 1);
    ::setenv("VGPU_JIT_CACHE", (dir / "cache").c_str(), 1);
    ::setenv("VGPU_QUIET", "1", 1);
    ::unsetenv("VGPU_TIER");
  }
  ~EnvInit() { std::error_code ec; fs::remove_all(dir, ec); }
} g_env;

void dummy_handler(void*, const tier::MOp&, uint32_t) {}

// A section of `kinds`: 'g' an op the generated code computes (a MOV of a constant), 'x' an op with no handler (it ends a region). `salt` makes its words
// (and so its cache key) its own.
struct FakeProgram {
  std::vector<Instr> instrs;
  tier::Program prog;
  FakeProgram(const std::string& kinds, uint64_t salt) {
    instrs.resize(kinds.size());
    prog.ops.resize(kinds.size());
    prog.pool.resize(1);
    for (size_t i = 0; i < kinds.size(); ++i) {
      Instr& in = instrs[i];
      in.w.lo = 0x3333000000000000ull + (salt << 16) + i;
      in.w.hi = 0x4444;
      in.op = Op::IADD3;
      tier::MOp& op = prog.ops[i];
      op.ins = &instrs[i];
      op.straight = true;
      op.fn = kinds[i] == 'x' ? nullptr : &dummy_handler;
      op.form = kinds[i] == 'g' ? tier::Form::Mov : tier::Form::None;
      op.unguarded = true;
      op.maxreg = static_cast<uint16_t>(4 + i);
    }
  }
};

int state_of(const jit::Unit& u) { return u.state.load(); }
constexpr int kCounting = static_cast<int>(jit::State::Counting), kWaiting = static_cast<int>(jit::State::Waiting), kIdle = static_cast<int>(jit::State::Idle);

}  // namespace

VTEST(jit_a_section_far_from_its_threshold_waits_on_the_plain_loop_and_counts_again_before_the_compile) {
  if (!jit::requested()) return;   // (a sanitizer build: the JIT is off)
  const uint64_t waits0 = jit::stats().waits.load(), ends0 = jit::stats().wait_ends.load(), compiles0 = jit::stats().compiles.load(),
                 fails0 = jit::stats().failures.load();
  FakeProgram fp("ggggggggx", 1);
  const auto u = jit::make_unit(fp.prog, 80);
  VCHECK(u != nullptr);
  VCHECK_EQ(u->regions.size(), size_t{1});
  VCHECK_EQ(state_of(*u), kIdle);
  VCHECK(u->hook());
  u->note(0, 100);
  VCHECK_EQ(state_of(*u), kIdle);   // 100 of 128: the cache is not asked yet
  u->note(0, 28);
  if (jit::stats().failures.load() != fails0) {   // (no compiler on PATH: the lookup cannot make a key, the section has failed and runs on the plain loop)
    std::printf("  (no C++ compiler on PATH: skipped)\n");
    return;
  }
  // 128: the cache asked, a miss; the threshold is 99,872 instructions away, more than twice the profile: the section waits.
  VCHECK_EQ(state_of(*u), kWaiting);
  VCHECK(u->waiting());
  VCHECK(!u->hook());       // the plain loop runs it
  VCHECK(!u->counting());
  VCHECK_EQ(jit::stats().waits.load(), waits0 + 1);
  VCHECK_EQ(u->wait_for.load(), kThreshold - kLookup - kProfile);   // ... until the threshold is one profile away
  u->note(0, 5000000);      // (an entry counted by a thread that was still in the hook loop: nothing)
  VCHECK_EQ(state_of(*u), kWaiting);
  VCHECK_EQ(jit::stats().compiles.load(), compiles0);
  u->wait_note(40000);
  u->wait_note(kThreshold - kLookup - kProfile - 40000 - 1);
  VCHECK_EQ(state_of(*u), kWaiting);   // one instruction short
  VCHECK_EQ(jit::stats().wait_ends.load(), ends0);
  u->wait_note(1);
  VCHECK_EQ(state_of(*u), kCounting);  // the wait is over: the hook loop again
  VCHECK(u->hook());
  VCHECK(u->counting());
  VCHECK(!u->waiting());
  VCHECK_EQ(jit::stats().wait_ends.load(), ends0 + 1);
  VCHECK_EQ(jit::stats().compiles.load(), compiles0);
  // The last stretch: the regions' counts go on from the 128 of before the lookup, and the wait counts for what it was set to. (Unit::note works the
  // section's total out when a region's count passes a multiple of 128 here, so the steps are multiples of it.)
  u->note(0, kProfile - 128);   // 10,000 counted, 89,872 waited: 128 short of the threshold
  VCHECK_EQ(state_of(*u), kCounting);
  VCHECK(!u->ready());
  VCHECK_EQ(jit::stats().compiles.load(), compiles0);
  u->note(0, 128);              // the threshold: compiled (synchronous mode) with the counts of both hooked stretches
  VCHECK(u->ready());
  VCHECK_EQ(jit::stats().compiles.load(), compiles0 + 1);
  VCHECK(!u->waiting());
  u->wait_note(1000000);        // (a thread that was still on the plain loop: nothing)
  VCHECK(u->ready());
  // One call of the plain loop may run past the end of the wait: the wait still counts for what it was set to, not for what the call ran.
  FakeProgram fq("ggggggggx", 2);
  const auto v = jit::make_unit(fq.prog, 80);
  VCHECK(v != nullptr);
  v->note(0, kLookup);
  VCHECK_EQ(state_of(*v), kWaiting);
  v->wait_note(5 * kThreshold);
  VCHECK_EQ(state_of(*v), kCounting);
  v->note(0, kProfile - 128);
  VCHECK_EQ(state_of(*v), kCounting);
  VCHECK(!v->ready());
  v->note(0, 128);
  VCHECK(v->ready());
  VCHECK_EQ(jit::stats().compiles.load(), compiles0 + 2);
}

VTEST(jit_a_section_near_its_threshold_at_the_lookup_counts_on_in_the_hook_loop) {
  if (!jit::requested()) return;
  const uint64_t waits0 = jit::stats().waits.load(), fails0 = jit::stats().failures.load();
  FakeProgram fp("ggggggggx", 3);
  const auto u = jit::make_unit(fp.prog, 80);
  VCHECK(u != nullptr);
  u->note(0, kThreshold - 2 * kProfile);   // the lookup with exactly twice the profile to go: no wait
  if (jit::stats().failures.load() != fails0) return;   // (no compiler)
  VCHECK_EQ(state_of(*u), kCounting);
  VCHECK(u->hook());
  VCHECK_EQ(jit::stats().waits.load(), waits0);
  FakeProgram fq("ggggggggx", 4);
  const auto v = jit::make_unit(fq.prog, 80);
  VCHECK(v != nullptr);
  v->note(0, kThreshold - 2 * kProfile - 1);   // one more to go than that: it waits
  VCHECK_EQ(state_of(*v), kWaiting);
  VCHECK_EQ(v->wait_for.load(), kProfile + 1);
  VCHECK_EQ(jit::stats().waits.load(), waits0 + 1);
}

VTEST_MAIN
