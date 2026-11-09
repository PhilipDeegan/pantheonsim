// What the SASS tier's run loop counts and when it stops must be what the interpreter's does, op by op (Runner::tier_run_t in src/sass/tier_ops.inc;
// the interpreter is VGPU_FASTPATH=0: one call of step_warp per instruction group). The places in that loop that work out the lanes an instruction
// counts, move the lanes' addresses and stop the run:
//   * the lanes a guard leaves (@P0, @!P0, none) and their count: a run of ops that share a predicate, a predicate redefined between two of them, a
//     guard that leaves no lane, all of them; the instruction, thread-instruction and per-class counters;
//   * loops whose lanes leave at different trips: the group (and its lane count) changes from call to call of the chain;
//   * `@P0 EXIT` for some lanes in the middle of a run (the lanes that stay go on to the next instruction: tier_set_pcs of the lanes that did not
//     leave), a split that leaves the high lanes alone, and an op that faults in the middle of a run;
//   * the step budget, ended at every offset of a run: with a budget of N, an instruction N + 1 must throw ExecLimit, in the tier's straight stretches
//     (which add their length at once) and in its one-op path alike, the same message as the interpreter's, and a budget of the whole run must not.
// Every kernel is run with the tier and with VGPU_FASTPATH=0; words, counters (every word before inst_by_opcode, so the classes too) and, for a
// launch that fails, the error must be the same.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/sass/exec.hpp"
#include "vgpu/sass/sass.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

struct Insn {
  sass::Word w;
  std::string text;   // empty: not checked against the decoder (the encodings that the test only needs to behave, not to read back)
};

std::string reg(unsigned r) { return r == 255 ? "RZ" : "R" + std::to_string(r); }
std::string hex(uint32_t v) {
  char b[16];
  std::snprintf(b, sizeof b, "0x%x", v);
  return b;
}
Insn make(uint64_t hi, uint64_t lo, std::string text) {
  Insn i;
  i.w.hi = hi, i.w.lo = lo;
  i.text = std::move(text);
  return i;
}
// lo: opcode | rd << 16 | ra << 24 | rb or immediate << 32; hi's low byte is the third register of the forms that have one.
uint64_t lo_of(uint64_t op, unsigned rd, unsigned ra, uint64_t b) { return op | uint64_t{rd} << 16 | uint64_t{ra} << 24 | b << 32; }

constexpr unsigned kRZ = 255;
// guard: 7 none (PT), 0 @P0, 8 @!P0
std::string guard_text(unsigned guard) { return guard == 7 ? "" : guard == 8 ? "@!P0 " : "@P0 "; }
Insn iadd3i(unsigned d, unsigned a, uint32_t imm, unsigned c, unsigned guard = 7) {
  return make(0x000fc80007ffe000ull | c, lo_of(0x0810 | (guard == 7 ? 0x7000 : guard << 12), d, a, imm),
              guard_text(guard) + "IADD3 " + reg(d) + ", " + reg(a) + ", " + hex(imm) + ", " + reg(c));
}
Insn imadi(unsigned d, unsigned a, uint32_t imm, unsigned c, unsigned guard = 7) {
  return make(0x000fe400078e0200ull | c, lo_of(0x0824 | (guard == 7 ? 0x7000 : guard << 12), d, a, imm),
              guard_text(guard) + "IMAD " + reg(d) + ", " + reg(a) + ", " + hex(imm) + ", " + reg(c));
}
Insn lop3(unsigned d, unsigned a, unsigned b, unsigned c, unsigned lut) {
  return make(0x000fe200078e0000ull | lut << 8 | c, lo_of(0x7212, d, a, b),
              "LOP3.LUT " + reg(d) + ", " + reg(a) + ", " + reg(b) + ", " + reg(c) + ", " + hex(lut) + ", !PT");
}
Insn movi(unsigned d, uint32_t imm) { return make(0x000fe20000000f00ull, lo_of(0x7802, d, 0, imm), "MOV " + reg(d) + ", " + hex(imm)); }
Insn isetp_ge(unsigned a, uint32_t imm) {   // P0 = (Ra >= imm), signed
  return make(0x000fc40003f06270ull, lo_of(0x780c, 0, a, imm), "ISETP.GE.AND P0, PT, " + reg(a) + ", " + hex(imm) + ", PT");
}
Insn s2r_tid(unsigned d) { return make(0x000e220000002100ull, lo_of(0x7919, d, 0, 0), "S2R " + reg(d) + ", SR_TID.X"); }
Insn exit_() { return make(0x000fea0003800000ull, 0x000000000000794dull, "EXIT"); }
Insn exit_guarded(unsigned guard) { return make(0x000fea0003800000ull, 0x000000000000094dull | (guard == 7 ? 0x7000 : uint64_t{guard} << 12), ""); }
// STG.E [Ra.64 + off], Rk
Insn stg(unsigned a, unsigned k, unsigned off) {
  return make(0x000fe2000c101904ull, lo_of(0x7986, 0, a, k) | uint64_t{off} << 40,
              "STG.E [" + reg(a) + ".64" + (off ? "+" + hex(off) : "") + "], " + reg(k));
}
// LDG.E Rd, [Ra.64]
Insn ldg(unsigned d, unsigned a) { return make(0x000ea2000c1e1900ull, lo_of(0x7981, d, a, 4), "LDG.E " + reg(d) + ", [" + reg(a) + ".64]"); }
// IMAD.WIDE.U32 Rd, Ra, imm, Rc
Insn imad_wide_i(unsigned d, unsigned a, uint32_t imm, unsigned c) {
  return make(0x000fe200078e0000ull | c, lo_of(0x7825, d, a, imm),
              "IMAD.WIDE.U32 " + reg(d) + ", " + reg(a) + ", " + hex(imm) + ", " + reg(c));
}
// @P0 BRA to the instruction at index `to`, from the one at index `from` (the offset is from the next instruction)
Insn bra_p0(int from, int to) {
  const int64_t off = int64_t{16} * (to - from - 1);
  const uint64_t hi = 0x000fc00003800000ull | (off < 0 ? 0x3ffffull : 0ull);
  char buf[32];
  std::snprintf(buf, sizeof buf, "@P0 BRA 0x%llx", static_cast<unsigned long long>(16 * to));
  return make(hi, 0x0947ull | (static_cast<uint64_t>(off) & 0xffffffffull) << 32, buf);
}

template <class T>
void put(std::string& s, T v) {
  s.append(reinterpret_cast<const char*>(&v), sizeof v);
}

// A cubin for sm_86 with one kernel taking one 8-byte pointer.
std::string make_cubin(const std::vector<Insn>& code) {
  const std::string names("\0.shstrtab\0.text.k\0.nv.info.k\0", 31);
  std::string info;   // KPARAM_INFO: index, ordinal | offset << 16, size << 18
  put<uint8_t>(info, 4), put<uint8_t>(info, 0x17), put<uint16_t>(info, 12);
  put<uint32_t>(info, 0), put<uint32_t>(info, 0), put<uint32_t>(info, 8u << 18);
  std::string out(64, '\0');
  const uint64_t names_off = out.size();
  out += names;
  while (out.size() % 16) out.push_back('\0');
  const uint64_t text_off = out.size();
  for (const Insn& i : code) put(out, i.w.lo), put(out, i.w.hi);
  const uint64_t info_off = out.size();
  out += info;
  while (out.size() % 8) out.push_back('\0');
  const uint64_t shoff = out.size();
  const auto shdr = [&](uint32_t name, uint32_t type, uint64_t off, uint64_t size) {
    put<uint32_t>(out, name), put<uint32_t>(out, type);
    put<uint64_t>(out, 0), put<uint64_t>(out, 0), put<uint64_t>(out, off), put<uint64_t>(out, size);
    put<uint32_t>(out, 0), put<uint32_t>(out, 0), put<uint64_t>(out, 16), put<uint64_t>(out, 0);
  };
  shdr(0, 0, 0, 0);
  shdr(1, 3, names_off, names.size());
  shdr(11, 1, text_off, 16 * code.size());
  shdr(19, 0x70000000, info_off, info.size());
  out[0] = 0x7f, out[1] = 'E', out[2] = 'L', out[3] = 'F', out[4] = 2, out[5] = 1, out[6] = 1, out[7] = 0x41, out[8] = 8;
  const uint16_t type = 2, machine = 190, shentsize = 64, shnum = 4, shstrndx = 1;
  const uint32_t flags = 0x06000002u | (86u << 8);
  std::memcpy(&out[16], &type, 2), std::memcpy(&out[18], &machine, 2);
  std::memcpy(&out[0x28], &shoff, 8), std::memcpy(&out[0x30], &flags, 4);
  std::memcpy(&out[0x3a], &shentsize, 2), std::memcpy(&out[0x3c], &shnum, 2), std::memcpy(&out[0x3e], &shstrndx, 2);
  return out;
}

struct Rng {
  uint64_t s = 0x9e3779b97f4a7c15ull;
  unsigned next(unsigned n) {
    s ^= s << 13, s ^= s >> 7, s ^= s << 17;
    return static_cast<unsigned>((s >> 11) % n);
  }
};

constexpr uint32_t kStride = 0x20;   // bytes a thread's results take (R1 .. R7, and its thread number)

std::vector<Insn> prologue() {
  std::vector<Insn> c;
  c.push_back(s2r_tid(0));
  c.push_back(make(0x000fc60000000f00ull, 0x0000580000007a02ull | uint64_t{12} << 16, "MOV R12, c[0x0][0x160]"));
  c.push_back(make(0x000fc60000000f00ull, 0x0000590000007a02ull | uint64_t{13} << 16, "MOV R13, c[0x0][0x164]"));
  c.push_back(imad_wide_i(14, 0, kStride, 12));
  for (unsigned r = 1; r <= 7; ++r) {   // values that differ from lane to lane and register to register
    c.push_back(imadi(r, 0, (0x9e3779b1u * r + 17) & 0x7fffffffu, kRZ));
    c.push_back(iadd3i(r, r, 0x1234567u * r, kRZ));
  }
  return c;
}
void epilogue(std::vector<Insn>& c) {
  for (unsigned r = 1; r <= 7; ++r) c.push_back(stg(14, r, 4 * r));
  c.push_back(stg(14, 0, 0));
  c.push_back(exit_());
}

// Branch-free: `groups` times, a compare on the thread number sets P0, and a few ops run under @P0, @!P0 or no guard, sharing the predicate, with a
// second compare in the middle of some of the groups (the lanes a guard leaves change under the same guard). A compare with a bound of 0 leaves every
// lane for @P0 and none for @!P0, and one of 99 the other way round.
std::vector<Insn> guarded_kernel(uint64_t seed, unsigned groups) {
  Rng rng;
  rng.s ^= (seed + 1) * 0xff51afd7ed558ccdull;
  std::vector<Insn> c = prologue();
  static const unsigned bounds[] = {0, 1, 7, 16, 16, 20, 31, 33, 40, 99};
  for (unsigned g = 0; g < groups; ++g) {
    c.push_back(isetp_ge(0, bounds[rng.next(10)]));
    const unsigned n = 1 + rng.next(7);
    for (unsigned i = 0; i < n; ++i) {
      if (rng.next(5) == 0) c.push_back(isetp_ge(0, bounds[rng.next(10)]));
      const unsigned guard = rng.next(4) == 0 ? 7u : (rng.next(2) ? 0u : 8u);
      const unsigned d = 1 + rng.next(7), a = rng.next(8);
      if (rng.next(3) == 0) c.push_back(imadi(d, a, rng.next(1u << 16) + 1, kRZ, guard));
      else c.push_back(iadd3i(d, a, rng.next(1000) + 1, rng.next(2) ? kRZ : 1 + rng.next(7), guard));
    }
  }
  epilogue(c);
  return c;
}

// Loops whose lanes leave at different trips: R5 = tid & 7 is the trip count of a lane; the lanes that have done their trips fall through to the
// stores, the others branch back (a backward branch that splits the group: the lowest address, the lanes still looping, runs first), so that the
// group the chain picks, and its lane count, change from call to call. Guarded ops in the loop body share P0 with the branch.
std::vector<Insn> loop_kernel(unsigned mask, unsigned guarded) {
  std::vector<Insn> c = prologue();
  c.push_back(movi(6, mask));
  c.push_back(lop3(5, 0, 6, kRZ, 0xc0));   // R5 = tid & mask
  const int top = static_cast<int>(c.size());
  c.push_back(iadd3i(1, 1, 3, kRZ));
  c.push_back(isetp_ge(5, 1));
  for (unsigned i = 0; i < guarded; ++i) c.push_back(iadd3i(2 + i % 3, 2 + i % 3, 7 + i, kRZ, i % 2 ? 8 : 0));
  Insn dec = iadd3i(5, 5, 0xffffffffu, kRZ);
  dec.text = "IADD3 R5, R5, -0x1, RZ";   // (the decoder prints a negative immediate with its sign)
  c.push_back(dec);
  c.push_back(isetp_ge(5, 1));
  c.push_back(bra_p0(static_cast<int>(c.size()), top));
  epilogue(c);
  return c;
}

// The lanes of tid >= `ge` branch over three ops that the others run (a split that leaves the low lanes together and the high ones, down to one lane,
// at another address: the chain picks the group at the lowest address, the low lanes, and then the high ones), with guarded ops on both sides.
std::vector<Insn> high_kernel(unsigned ge) {
  std::vector<Insn> c = prologue();
  c.push_back(isetp_ge(0, ge));
  const int from = static_cast<int>(c.size());
  c.push_back(bra_p0(from, from + 4));
  c.push_back(iadd3i(1, 1, 1, kRZ));
  c.push_back(iadd3i(2, 2, 3, kRZ, 8));
  c.push_back(iadd3i(3, 3, 5, kRZ));
  c.push_back(iadd3i(4, 4, 7, kRZ, 0));
  c.push_back(iadd3i(5, 5, 9, kRZ));
  epilogue(c);
  return c;
}

// Lanes of tid >= `ge` leave in the middle of a run (`@P0 EXIT`, or `@!P0 EXIT` with `negate`); the others go on with guarded ops and store.
std::vector<Insn> exit_kernel(unsigned ge, bool negate) {
  std::vector<Insn> c = prologue();
  c.push_back(iadd3i(1, 1, 5, kRZ));
  c.push_back(isetp_ge(0, ge));
  c.push_back(iadd3i(2, 2, 9, kRZ, 0));
  c.push_back(exit_guarded(negate ? 8 : 0));
  c.push_back(iadd3i(3, 3, 11, kRZ, 8));
  c.push_back(iadd3i(4, 4, 13, kRZ));
  c.push_back(isetp_ge(0, ge / 2));
  c.push_back(iadd3i(5, 5, 17, kRZ, 0));
  epilogue(c);
  return c;
}

// A run of ops, then a load from an address that no allocation holds: the load faults, and the fault names it.
std::vector<Insn> fault_kernel(unsigned before) {
  std::vector<Insn> c = prologue();
  for (unsigned i = 0; i < before; ++i) c.push_back(iadd3i(1 + i % 7, 1 + i % 7, i + 1, kRZ, i % 3 == 0 ? 0 : 7));
  c.push_back(isetp_ge(0, 16));
  c.push_back(iadd3i(2, 2, 1, kRZ, 0));
  c.push_back(movi(14, 0x10));
  c.push_back(movi(15, 0));
  c.push_back(ldg(1, 14));
  c.push_back(iadd3i(1, 1, 1, kRZ));
  epilogue(c);
  return c;
}

// Straight-line, longer than the 256 ops the loop takes at a time, with guarded ops now and then.
std::vector<Insn> long_kernel(unsigned body) {
  std::vector<Insn> c = prologue();
  c.push_back(isetp_ge(0, 16));
  for (unsigned i = 0; i < body; ++i)
    c.push_back(iadd3i(1 + i % 7, 1 + (i + 3) % 7, i + 1, kRZ, i % 11 == 5 ? 0u : (i % 13 == 7 ? 8u : 7u)));
  epilogue(c);
  return c;
}

struct Outcome {
  bool ok = false;
  std::string error;   // the message, when the launch threw
  int code = 0;
  std::vector<uint32_t> out;
  exec::LaunchStats stats;
};

struct Env {
  MemoryManager mem{1 << 24};
  DeviceProfile prof = load_gpu("nvidia/a10");   // sm_86
  std::shared_ptr<sass::Module> load(const std::vector<Insn>& code) {
    for (size_t i = 0; i < code.size(); ++i) {
      const sass::Instr ins = sass::decode(code[i].w, 16 * i, 86);
      if (!code[i].text.empty() && sass::to_text(ins) != code[i].text)
        throw vtest::Failure("word " + std::to_string(i) + " is \"" + sass::to_text(ins) + "\", not \"" + code[i].text + "\"");
    }
    const std::string c = make_cubin(code);
    return sass::load(reinterpret_cast<const uint8_t*>(c.data()), c.size(), mem, prof);
  }
  Outcome run(sass::Module& m, uint32_t threads, uint64_t max_steps = 1ull << 30) {
    const uint32_t words = threads * kStride / 4;
    const uint64_t out = mem.alloc(words * 4);
    const std::vector<uint32_t> poison(words, 0x5A5A5A5Au);
    mem.write(out, poison.data(), words * 4);
    exec::LaunchConfig cfg;
    cfg.grid = {1, 1, 1};
    cfg.block = {threads, 1, 1};
    cfg.max_steps = max_steps;
    std::vector<uint8_t> arg(8);
    std::memcpy(arg.data(), &out, 8);
    Outcome r;
    try {
      r.stats = sass::launch(m, "k", cfg, {arg}, mem, prof);
      r.ok = true;
    } catch (const Error& e) {
      r.error = e.message();
      r.code = static_cast<int>(e.code());
    }
    r.out.resize(words);
    mem.read(out, r.out.data(), words * 4);
    mem.free(out);
    return r;
  }
};

struct Scoped {
  Scoped() { ::setenv("VGPU_THREADS", "1", 1); }
  ~Scoped() {
    ::unsetenv("VGPU_THREADS");
    ::unsetenv("VGPU_FASTPATH");
  }
};

void expect_same(const Outcome& tier, const Outcome& ref, const std::string& what) {
  if (tier.ok != ref.ok || tier.error != ref.error || tier.code != ref.code)
    throw vtest::Failure(what + ": " + (tier.ok ? "the tier's launch ran" : "the tier's launch threw \"" + tier.error + "\"") + ", " +
                         (ref.ok ? "the interpreter's ran" : "the interpreter's threw \"" + ref.error + "\""));
  if (tier.out != ref.out) {
    size_t i = 0;
    while (i < tier.out.size() && tier.out[i] == ref.out[i]) ++i;
    throw vtest::Failure(what + ": word " + std::to_string(i) + " (thread " + std::to_string(i / (kStride / 4)) + ", register " + std::to_string(i % (kStride / 4)) + ") is " +
                         hex(tier.out[i]) + " with the tier and " + hex(ref.out[i]) + " without");
  }
  if (!tier.ok) return;
  if (std::memcmp(&tier.stats, &ref.stats, exec::kCounterWords * sizeof(uint64_t)) != 0) {
    const uint64_t* a = reinterpret_cast<const uint64_t*>(&tier.stats);
    const uint64_t* b = reinterpret_cast<const uint64_t*>(&ref.stats);
    size_t i = 0;
    while (a[i] == b[i]) ++i;
    throw vtest::Failure(what + ": counter word " + std::to_string(i) + " is " + std::to_string(a[i]) + " with the tier and " + std::to_string(b[i]) + " without");
  }
}

// The kernel run by the tier and by the interpreter at one block size and one step budget; the tier's outcome.
Outcome compare(Env& e, sass::Module& m, uint32_t threads, uint64_t max_steps, const std::string& name) {
  const Outcome tier = e.run(m, threads, max_steps);
  ::setenv("VGPU_FASTPATH", "0", 1);
  const Outcome ref = e.run(m, threads, max_steps);
  ::unsetenv("VGPU_FASTPATH");
  expect_same(tier, ref, name + ", " + std::to_string(threads) + " threads, budget " + std::to_string(max_steps));
  return tier;
}

Outcome compare(const std::vector<Insn>& code, std::initializer_list<uint32_t> sizes, const std::string& name) {
  Scoped scoped;
  Env e;
  const auto m = e.load(code);
  Outcome last;
  for (uint32_t threads : sizes) last = compare(e, *m, threads, 1ull << 30, name);
  return last;
}

}  // namespace

VTEST(guarded_runs_count_the_lanes_the_interpreter_counts) {
  // (every compare of a launch of 32 threads is a full warp; 40 is a warp and a part of one; 20 a part of one: their groups never are the whole warp)
  for (uint64_t seed = 0; seed < 60; ++seed)
    compare(guarded_kernel(seed, 3 + static_cast<unsigned>(seed % 9)), {32, 40, 20, 64}, "guarded kernel " + std::to_string(seed));
}

VTEST(loops_that_split_the_group_count_the_lanes_of_every_group) {
  for (unsigned mask : {3u, 7u, 15u})
    for (unsigned guarded : {0u, 1u, 2u, 4u}) {
      const std::string name = "loop mask " + std::to_string(mask) + ", " + std::to_string(guarded) + " guarded";
      const Outcome o = compare(loop_kernel(mask, guarded), {32, 40, 96, 20}, name);
      VCHECK(o.ok);
      VCHECK(o.stats.divergent_branches > 0);   // (the branch does split the group: the test is not vacuous)
    }
}

VTEST(a_split_that_leaves_the_high_lanes_alone_picks_the_low_group_first) {
  for (unsigned ge : {1u, 8u, 16u, 20u, 24u, 28u, 31u}) {
    const Outcome o = compare(high_kernel(ge), {32, 40, 64}, "high lanes from " + std::to_string(ge));
    VCHECK(o.ok);
    VCHECK(o.stats.divergent_branches > 0);
  }
}

VTEST(lanes_that_exit_in_the_middle_of_a_run_are_counted_up_to_the_exit) {
  for (unsigned ge : {0u, 1u, 8u, 16u, 20u, 31u, 33u, 99u})
    for (bool negate : {false, true}) compare(exit_kernel(ge, negate), {32, 40, 64}, "exit at " + std::to_string(ge) + (negate ? " negated" : ""));
}

VTEST(an_op_that_faults_in_a_run_is_the_one_the_interpreter_names) {
  Scoped scoped;
  Env e;
  for (unsigned before : {0u, 1u, 5u, 40u, 300u}) {
    const auto m = e.load(fault_kernel(before));
    for (uint32_t threads : {32u, 40u}) {
      const Outcome o = compare(e, *m, threads, 1ull << 30, "fault after " + std::to_string(before) + " ops");
      VCHECK(!o.ok);   // (it does fault)
      VCHECK(!o.error.empty());
    }
  }
}

VTEST(a_step_budget_ends_a_run_at_every_offset_where_the_interpreter_ends_it) {
  Scoped scoped;
  Env e;
  // One warp, so the warp's steps are the instructions the launch counts. A budget of N lets N instructions run and the next one throw: the launch
  // fails for every N below the number of instructions and runs for every N from it, in the tier as in the interpreter, at every offset.
  const struct {
    const char* name;
    std::vector<Insn> code;
  } kernels[] = {
      {"guarded", guarded_kernel(5, 5)},
      {"loop", loop_kernel(3, 2)},
      {"exit", exit_kernel(16, false)},
      {"long", long_kernel(300)},   // (a straight stretch of more than 256 ops: the loop cuts it at 256)
  };
  for (const auto& k : kernels) {
    const auto m = e.load(k.code);
    ::setenv("VGPU_FASTPATH", "0", 1);
    const Outcome whole = e.run(*m, 32);
    ::unsetenv("VGPU_FASTPATH");
    VCHECK(whole.ok);
    const uint64_t total = whole.stats.instructions;
    VCHECK(total > 25);
    for (uint64_t n = 1; n <= total + 3; ++n) {
      const Outcome o = compare(e, *m, 32, n, k.name);
      if (o.ok != (n >= total))
        throw vtest::Failure(std::string(k.name) + ": with a budget of " + std::to_string(n) + " of " + std::to_string(total) + " steps the launch " + (o.ok ? "ran" : "threw"));
    }
    // And with several warps (each has the budget of its own): the same offsets, a part of a warp too.
    for (uint64_t n : {total - 1, total, total + 1}) compare(e, *m, 96, n, k.name);
  }
}

VTEST_MAIN
