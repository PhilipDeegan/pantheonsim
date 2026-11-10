// VGPU_EXACT_THREADS (docs/exact-threads.md): with VGPU_THREADS=1 a launch gives the bits of its blocks run in order, and with
// VGPU_EXACT_THREADS=N beside it the SASS executor runs the blocks on N host threads on trial, under a shadow of device memory
// (MemoryManager::spec_*), and keeps the trial only if no block touched what another wrote. These tests hold it to that:
//   * kernels whose blocks are independent commit, and leave the memory and every counter of the run in order;
//   * kernels whose blocks write the same bytes, or read what an earlier block wrote, are stopped, undone and run in order, and
//     leave the memory and the counters of the run in order (which a parallel run without the check does not);
//   * a kernel that faults in block 1 leaves exactly what the run in order leaves, the later blocks' stores undone, and reports the same error;
//   * a kernel that reads its SM number on a grid larger than the device is stopped (the number is the host thread's);
//   * every method of MemoryManager that reads or writes device bytes is noted by the shadow (a method added without its hook
//     would let a trial commit what it should not), and the shadow is empty again after a trial.
// The kernels are SASS for sm_86 written out word by word; nvdisasm's text of each word is checked against the decoder's.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
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
Insn s2r_tid(unsigned d) { return make(0x000e220000002100ull, lo_of(0x7919, d, 0, 0), "S2R " + reg(d) + ", SR_TID.X"); }
Insn exit_() { return make(0x000fea0003800000ull, 0x000000000000794dull, "EXIT"); }
// STG.E [Ra.64 + off], Rk
Insn stg(unsigned a, unsigned k, unsigned off) {
  return make(0x000fe2000c101904ull, lo_of(0x7986, 0, a, k) | uint64_t{off} << 40,
              "STG.E [" + reg(a) + ".64" + (off ? "+" + hex(off) : "") + "], " + reg(k));
}
// IMAD.WIDE.U32 Rd, Ra, imm, Rc
Insn imad_wide_i(unsigned d, unsigned a, uint32_t imm, unsigned c) {
  return make(0x000fe200078e0000ull | c, lo_of(0x7825, d, a, imm),
              "IMAD.WIDE.U32 " + reg(d) + ", " + reg(a) + ", " + hex(imm) + ", " + reg(c));
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

Insn s2r_ctaid(unsigned d) { return make(0x000ea20000002500ull, lo_of(0x7919, d, 0, 0), "S2R " + reg(d) + ", SR_CTAID.X"); }
Insn s2r_smid(unsigned d) { return make(0x000e220000004300ull, lo_of(0x7919, d, 0, 0), "S2R " + reg(d) + ", SR_VIRTUALSMID"); }
Insn iadd3(unsigned d, unsigned a, unsigned b, unsigned c) {
  return make(0x000fc80007ffe000ull | c, lo_of(0x7210, d, a, b), "IADD3 " + reg(d) + ", " + reg(a) + ", " + reg(b) + ", " + reg(c));
}
// LDG.E Rd, [Ra.64 + off]
Insn ldg_off(unsigned d, unsigned a, unsigned off) {
  return make(0x000ea2000c1e1900ull, lo_of(0x7981, d, a, 4) | uint64_t{off} << 40,
              "LDG.E " + reg(d) + ", [" + reg(a) + ".64" + (off ? "+" + hex(off) : "") + "]");
}
const Insn kPtrLo = make(0x000fc60000000f00ull, 0x0000580000067a02ull, "MOV R6, c[0x0][0x160]");
const Insn kPtrHi = make(0x000fc60000000f00ull, 0x0000590000077a02ull, "MOV R7, c[0x0][0x164]");

constexpr uint32_t kThreads = 64;        // two warps a block
constexpr uint32_t kIn = 0x40000;        // the byte offset of the input half of the buffer (64 K words: more than any grid here writes)
constexpr uint32_t kWords = 2 * (kIn / 4);

// The start of every kernel: R0 = blockIdx.x, R1 = threadIdx.x, R6:R7 = the buffer, R2 = blockIdx.x * 64 + threadIdx.x, R10:R11 = &buf[R2].
std::vector<Insn> prologue() {
  return {s2r_ctaid(0), s2r_tid(1), kPtrLo, kPtrHi, imadi(2, 0, kThreads, 1), imad_wide_i(10, 2, 4, 6)};
}
// Independent blocks: buf[i] = in[i] + in[0], i = block * 64 + thread. Every block reads in[0]; each writes its own 64 words.
std::vector<Insn> independent() {
  std::vector<Insn> c = prologue();
  c.insert(c.end(), {ldg_off(3, 10, kIn), ldg_off(4, 6, kIn), iadd3(5, 3, 4, kRZ), stg(10, 5, 0), exit_()});
  return c;
}
// Every block writes buf[thread] = block: in order the last block's number stays.
std::vector<Insn> same_words() {
  std::vector<Insn> c = prologue();
  c.insert(c.end(), {imad_wide_i(12, 1, 4, 6), stg(12, 0, 0), exit_()});
  return c;
}
// A chain: buf[i + 64] = buf[i] + 1. Block b reads the words block b - 1 wrote: in order row b holds row 0 + b.
std::vector<Insn> chain() {
  std::vector<Insn> c = prologue();
  c.insert(c.end(), {ldg_off(3, 10, 0), iadd3i(3, 3, 1, kRZ), stg(10, 3, 4 * kThreads), exit_()});
  return c;
}
// buf[i] = block, then a store to buf + block * 2^28: far outside every allocation for block 1 and later (block 0 writes buf[0] again).
std::vector<Insn> faults_from_block_1() {
  std::vector<Insn> c = prologue();
  c.insert(c.end(), {stg(10, 0, 0), imad_wide_i(12, 0, 0x10000000, 6), stg(12, 0, 0), exit_()});
  return c;
}
// buf[i] = the SM number.
std::vector<Insn> sm_number() {
  std::vector<Insn> c = prologue();
  c.insert(c.end(), {s2r_smid(3), stg(10, 3, 0), exit_()});
  return c;
}

struct Result {
  std::vector<uint32_t> words;
  exec::LaunchStats stats;
  std::string error;   // what the launch threw ("" when it returned)
  sass::ExactCounts counts;   // of this launch alone
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
  // One launch of `blocks` blocks of 64 threads over a buffer whose words start as 3 * index + 7.
  Result run(sass::Module& m, uint32_t blocks) {
    const uint64_t buf = mem.alloc(uint64_t{kWords} * 4);
    std::vector<uint32_t> init(kWords);
    for (uint32_t i = 0; i < kWords; ++i) init[i] = 3 * i + 7;
    mem.write(buf, init.data(), uint64_t{kWords} * 4);
    exec::LaunchConfig cfg;
    cfg.grid = {blocks, 1, 1};
    cfg.block = {kThreads, 1, 1};
    std::vector<uint8_t> arg(8);
    std::memcpy(arg.data(), &buf, 8);
    Result r;
    const sass::ExactCounts before = sass::exact_counts();
    try {
      r.stats = sass::launch(m, "k", cfg, {arg}, mem, prof);
    } catch (const Error& e) {
      r.error = e.what();
      // (every run has a buffer of its own: the one address a message here names, buf + 2^28, is written the same for all)
      char far[32];
      std::snprintf(far, sizeof far, "0x%llx", static_cast<unsigned long long>(buf + 0x10000000ull));
      if (const size_t at = r.error.find(far); at != std::string::npos) r.error.replace(at, std::strlen(far), "buf+2^28");
    }
    const sass::ExactCounts after = sass::exact_counts();
    r.counts = {after.trials - before.trials, after.committed - before.committed, after.stopped - before.stopped};
    r.words.resize(kWords);
    mem.read(buf, r.words.data(), uint64_t{kWords} * 4);
    mem.free(buf);
    return r;
  }
};

void in_order() {
  ::setenv("VGPU_THREADS", "1", 1);
  ::unsetenv("VGPU_EXACT_THREADS");
}
void exact(const char* n) {
  ::setenv("VGPU_THREADS", "1", 1);
  ::setenv("VGPU_EXACT_THREADS", n, 1);
  ::setenv("VGPU_POOL_MIN_WORK_US", "0", 1);   // (the width rule keeps launches as small as these on one thread: off, so that every launch is tried)
}
struct Restore {
  ~Restore() {
    ::unsetenv("VGPU_THREADS");
    ::unsetenv("VGPU_EXACT_THREADS");
    ::unsetenv("VGPU_POOL_MIN_WORK_US");
  }
};

// Every counter of the two launches, the per-opcode ones too.
void same_counters(const exec::LaunchStats& a, const exec::LaunchStats& b, const std::string& what) {
  if (std::memcmp(static_cast<const void*>(&a), static_cast<const void*>(&b), exec::kCounterWords * sizeof(uint64_t)) != 0)
    throw vtest::Failure(what + ": the counters differ (instructions " + std::to_string(a.instructions) + " and " + std::to_string(b.instructions) + ")");
  if (a.inst_by_opcode != b.inst_by_opcode) throw vtest::Failure(what + ": the per-opcode counts differ");
}

// The kernel in order, then with VGPU_EXACT_THREADS = 2, 3, 4 and 8 (three launches each, and each on a module of its own so that
// no launch is held back by an earlier stop): the same words, the same counters, the same error, and the trial ending as `commits` says.
Result same_as_in_order(const std::vector<Insn>& code, uint32_t blocks, bool commits, const std::string& name) {
  Restore restore;
  Env e;
  in_order();
  const Result ref = e.run(*e.load(code), blocks);
  VCHECK_EQ(ref.counts.trials, uint64_t{0});
  for (const char* n : {"2", "3", "4", "8"}) {
    exact(n);
    for (int rep = 0; rep < 3; ++rep) {
      const Result got = e.run(*e.load(code), blocks);
      const std::string what = name + ", VGPU_EXACT_THREADS=" + n;
      if (got.words != ref.words) {
        size_t i = 0;
        while (got.words[i] == ref.words[i]) ++i;
        throw vtest::Failure(what + ": word " + std::to_string(i) + " is " + std::to_string(got.words[i]) + ", in order " + std::to_string(ref.words[i]));
      }
      if (got.error != ref.error) throw vtest::Failure(what + ": the error is \"" + got.error + "\", in order \"" + ref.error + "\"");
      if (ref.error.empty()) same_counters(got.stats, ref.stats, what);
      if (got.counts.trials != 1 || got.counts.committed != (commits ? 1u : 0u) || got.counts.stopped != (commits ? 0u : 1u))
        throw vtest::Failure(what + ": " + std::to_string(got.counts.trials) + " trials, " + std::to_string(got.counts.committed) + " committed, " +
                             std::to_string(got.counts.stopped) + " stopped");
    }
  }
  return ref;
}

}  // namespace

VTEST(independent_blocks_commit_and_leave_the_bits_and_counters_of_the_run_in_order) {
  const Result r = same_as_in_order(independent(), 300, /*commits=*/true, "independent");
  VCHECK(r.error.empty());
  // (and the run in order is the kernel's arithmetic: buf[i] = in[i] + in[0])
  const uint32_t in0 = 3 * (kIn / 4) + 7;
  for (uint32_t i = 0; i < 300 * kThreads; ++i) VCHECK_EQ(r.words[i], 3 * (kIn / 4 + i) + 7 + in0);
  VCHECK_EQ(r.words[300 * kThreads], 3 * (300 * kThreads) + 7);
  VCHECK_EQ(r.stats.blocks, uint64_t{300});
  VCHECK_EQ(r.stats.global_stores, uint64_t{300} * kThreads);
}

VTEST(blocks_that_write_the_same_words_are_stopped_and_run_in_order) {
  const Result r = same_as_in_order(same_words(), 300, /*commits=*/false, "same words");
  for (uint32_t t = 0; t < kThreads; ++t) VCHECK_EQ(r.words[t], 299u);   // the last block's number
}

VTEST(a_block_that_reads_what_an_earlier_block_wrote_is_stopped_and_run_in_order) {
  const Result r = same_as_in_order(chain(), 200, /*commits=*/false, "chain");
  for (uint32_t b = 1; b <= 200; ++b)
    for (uint32_t t = 0; t < kThreads; ++t) VCHECK_EQ(r.words[b * kThreads + t], 3 * t + 7 + b);   // row 0 plus b: each block saw the one before it
}

VTEST(a_launch_that_faults_leaves_what_the_run_in_order_leaves) {
  const Result r = same_as_in_order(faults_from_block_1(), 300, /*commits=*/false, "fault in block 1");
  VCHECK_CONTAINS(r.error, "device memory write at buf+2^28");
  // In order: block 0 ran, block 1 stored its words and then faulted, no later block ran. (A trial runs later blocks too: their words must be put back.)
  for (uint32_t i = 2 * kThreads; i < 300 * kThreads; ++i) VCHECK_EQ(r.words[i], 3 * i + 7);
}

VTEST(a_kernel_that_reads_its_sm_number_on_a_grid_larger_than_the_device_is_stopped) {
  const Result r = same_as_in_order(sm_number(), 300, /*commits=*/false, "SM number");   // (the a10 has 72 SMs)
  for (uint32_t i = 0; i < 300 * kThreads; ++i) VCHECK_EQ(r.words[i], 0u);   // in order every block is on host thread 0
}

VTEST(a_kernel_whose_trial_was_stopped_is_held_back_and_tried_again_later) {
  Restore restore;
  Env e;
  exact("4");
  const auto m = e.load(same_words());
  // Stopped, then in order for 1 launch, stopped, in order for 2, stopped, in order for 4: trials at launches 0, 2, 5, 10.
  std::vector<int> tried;
  for (int i = 0; i < 12; ++i)
    if (e.run(*m, 64).counts.trials) tried.push_back(i);
  VCHECK(tried == (std::vector<int>{0, 2, 5, 10}));
}

VTEST(nothing_is_tried_without_the_variable_or_without_blocks_in_order) {
  Restore restore;
  Env e;
  const auto m = e.load(independent());
  exact("4");
  ::unsetenv("VGPU_THREADS");   // (several threads the usual way: the variable asks for nothing)
  VCHECK_EQ(e.run(*m, 64).counts.trials, uint64_t{0});
  ::setenv("VGPU_THREADS", "4", 1);
  VCHECK_EQ(e.run(*m, 64).counts.trials, uint64_t{0});
  in_order();
  VCHECK_EQ(e.run(*m, 64).counts.trials, uint64_t{0});
  exact("1");
  VCHECK_EQ(e.run(*m, 64).counts.trials, uint64_t{0});
  exact("4");
  VCHECK_EQ(e.run(*m, 64).counts.trials, uint64_t{1});
  VCHECK_EQ(e.run(*m, 1).counts.trials, uint64_t{0});   // one block: nothing to run side by side
}

// ---- the shadow itself: every method that touches device bytes is noted ----

namespace {
struct Trial {
  MemoryManager& mem;
  explicit Trial(MemoryManager& m) : mem(m) {
    mem.spec_begin(2);
    mem.spec_enter(0);
    mem.spec_block(0);
  }
  ~Trial() {
    if (mem.spec_active()) end();
  }
  MemoryManager::SpecOutcome end() {
    mem.spec_leave();
    return mem.spec_end();
  }
};
bool stops(const std::function<void()>& f) {
  try {
    f();
  } catch (const MemoryManager::SpecStop&) {
    return true;
  }
  return false;
}
}  // namespace

VTEST(every_reading_method_is_stopped_at_bytes_another_block_wrote_and_every_writing_method_at_bytes_another_block_read) {
  MemoryManager mem{1 << 24};
  const uint64_t a = mem.alloc(1 << 16);
  std::vector<uint8_t> init(1 << 16);
  for (size_t i = 0; i < init.size(); ++i) init[i] = static_cast<uint8_t>(i * 7 + 1);
  mem.write(a, init.data(), init.size());
  uint64_t v[64] = {};
  uint64_t addrs[32];
  for (int i = 0; i < 32; ++i) addrs[i] = a + 0x100 + 8 * i;
  const uint8_t sizes[2] = {4, 4};
  uint8_t* ptrs[32];
  size_t done = 0;
  const auto same = +[](void*, uint64_t old) -> uint64_t { return old; };
  const auto lane = +[](void*, size_t, uint64_t old) -> uint64_t { return old; };
  // Block 0 writes the words at a + 0x100 .. a + 0x200; each reading method, called as block 1 on those bytes, must stop the trial
  // (or, for a method that declines during a trial, say no: its caller then takes a path that is noted).
  const std::pair<const char*, std::function<bool()>> readers[] = {
      {"read", [&] { return stops([&] { mem.read(a + 0x100, v, 8); }); }},
      {"load_scalar", [&] { return stops([&] { mem.load_scalar(a + 0x100, 4); }); }},
      {"load_run", [&] { return stops([&] { mem.load_run(a + 0x100, v, 128, 4); }); }},
      {"load_batch", [&] { return stops([&] { mem.load_batch(addrs, 32, 8, v); }); }},
      {"load_batch16", [&] { return stops([&] { mem.load_batch16(addrs, 4, v, v + 32); }); }},
      {"load_broadcast", [&] { return stops([&] { mem.load_broadcast(a + 0x100, 32, 4, v); }); }},
      {"load_quad", [&] { return stops([&] { mem.load_quad(a + 0x100, v); }); }},
      {"load_quad_run", [&] { return stops([&] { mem.load_quad_run(a + 0x100, 2, v); }); }},
      {"load_fields", [&] { return stops([&] { mem.load_fields(a + 0x100, sizes, 2, v); }); }},
      {"load_texels", [&] { return stops([&] { mem.load_texels(addrs, 4, 1, 4, v); }); }},
      {"batch_window", [&] { MemoryManager::BatchWindow w; return !mem.batch_window(addrs, 32, 8, &w); }},
  };
  for (const auto& [name, f] : readers) {
    Trial t(mem);
    mem.store_scalar(a + 0x100, 8, 1);
    for (int i = 1; i < 32; ++i) mem.store_scalar(addrs[i], 8, 1);
    mem.spec_block(1);
    if (!f()) throw vtest::Failure(std::string(name) + " read what another block wrote and the trial went on");
    const MemoryManager::SpecOutcome out = t.end();
    VCHECK(std::string(name) == "batch_window" ? out.committed : !out.committed);
    if (out.committed) mem.write(a, init.data(), init.size());   // (declined, not stopped: block 0's stores stayed)
  }
  // Block 0 reads them; each writing method, called as block 1, must stop the trial.
  const std::pair<const char*, std::function<bool()>> writers[] = {
      {"write", [&] { return stops([&] { mem.write(a + 0x100, v, 8); }); }},
      {"fill", [&] { const uint8_t z = 0; return stops([&] { mem.fill(a + 0x100, &z, 1, 8); }); }},
      {"store_scalar", [&] { return stops([&] { mem.store_scalar(a + 0x100, 4, 5); }); }},
      {"store_run", [&] { return stops([&] { mem.store_run(a + 0x100, v, 128, 4); }); }},
      {"store_batch", [&] { return stops([&] { mem.store_batch(addrs, 32, 8, v); }); }},
      {"store_strided", [&] { return stops([&] { mem.store_strided(a + 0x100, 8, 32, 8, v); }); }},
      {"batch_store", [&] { MemoryManager::BatchWindow w{}; return stops([&] { mem.batch_store(w, addrs, 32, 8, v); }); }},
      {"store_quad", [&] { return stops([&] { mem.store_quad(a + 0x100, v); }); }},
      {"store_quad_run", [&] { return stops([&] { mem.store_quad_run(a + 0x100, 2, v); }); }},
      {"rmw_scalar", [&] { return stops([&] { mem.rmw_scalar_fn(a + 0x100, 4, same, nullptr, v); }); }},
      {"rmw_if_changed", [&] { return stops([&] { mem.rmw_if_changed_fn(a + 0x100, 4, same, nullptr, v); }); }},
      {"atomic_rmw", [&] { return stops([&] { mem.atomic_rmw_fn(a + 0x100, 4, same, nullptr, v, false); }); }},
      {"rmw_batch", [&] { return stops([&] { mem.rmw_batch_fn(addrs, 32, 8, lane, nullptr, v, &done, false, false); }); }},
      {"atomic_word_ptrs", [&] { return !mem.atomic_word_ptrs(addrs, 32, 8, ptrs); }},
      {"copy_to", [&] { return stops([&] { mem.copy_to(mem, a + 0x100, a + 0x4000, 64); }); }},
  };
  for (const auto& [name, f] : writers) {
    Trial t(mem);
    mem.read(a + 0x100, v, 256);
    mem.spec_block(1);
    if (!f()) throw vtest::Failure(std::string(name) + " wrote what another block read and the trial went on");
    const MemoryManager::SpecOutcome out = t.end();
    VCHECK(std::string(name) == "atomic_word_ptrs" ? out.committed : !out.committed);
  }
  // What a trial may not do at all.
  const std::pair<const char*, std::function<void()>> never[] = {
      {"alloc", [&] { mem.alloc(64); }},
      {"free", [&] { mem.free(a); }},
      {"heap_alloc", [&] { mem.heap_alloc(64, 1 << 20); }},
      {"heap_free", [&] { mem.heap_free(a); }},
      {"intern_string", [&] { mem.intern_string("x"); }},
      {"a store outside the device's memory", [&] { mem.store_scalar(a + (uint64_t{1} << 40), 4, 1); }},
  };
  for (const auto& [name, f] : never) {
    Trial t(mem);
    if (!stops(f)) throw vtest::Failure(std::string(name) + " went on during a trial");
    VCHECK(!t.end().committed);
  }
  // Nothing above changed a byte: every trial that wrote was undone or wrote what was there... the stores of block 0 were undone.
  std::vector<uint8_t> now(init.size());
  mem.read(a, now.data(), now.size());
  VCHECK(now == init);
}

VTEST(blocks_may_share_what_they_only_read_and_keep_what_they_alone_write_and_a_stopped_trial_is_undone_to_the_byte) {
  MemoryManager mem{1 << 24};
  const uint64_t a = mem.alloc(3 * 65536 + 10);   // (three chunks and a tail that is no whole word)
  std::vector<uint8_t> init(3 * 65536 + 10);
  for (size_t i = 0; i < init.size(); ++i) init[i] = static_cast<uint8_t>(i * 13 + 5);
  mem.write(a + 100, init.data() + 100, init.size() - 100);   // (the first bytes stay as an allocation starts: never written)
  std::fill(init.begin(), init.begin() + 100, uint8_t{0});
  const auto all = [&] {
    std::vector<uint8_t> now(init.size());
    mem.read(a, now.data(), now.size());
    return now;
  };
  uint64_t v = 0;
  {
    // Two blocks read the same word, each writes words of its own (one of them a byte of the unwritten start, one the last bytes): committed.
    Trial t(mem);
    VCHECK_EQ(mem.load_scalar(a + 4096, 4), mem.load_scalar(a + 4096, 4));
    mem.store_scalar(a + 8, 1, 0xAB);
    mem.spec_block(1);
    mem.load_scalar(a + 4096, 4);
    mem.store_scalar(a + 3 * 65536 + 9, 1, 0xCD);
    mem.store_scalar(a + 12, 1, 0xEF);   // the next word after block 0's: its own
    const MemoryManager::SpecOutcome out = t.end();
    VCHECK(out.committed);
    VCHECK_EQ(out.kept_bytes, uint64_t{4 + 2 + 4});   // a word, the two bytes the allocation has of the last word, a word
    init[8] = 0xAB, init[3 * 65536 + 9] = 0xCD, init[12] = 0xEF;
    VCHECK(all() == init);
  }
  {
    // A block writes across two chunks and single bytes here and there, then another block touches one of its words: all of it goes back.
    Trial t(mem);
    std::vector<uint8_t> junk(70000, 0x77);
    mem.write(a + 65536 - 3000, junk.data(), junk.size());
    mem.store_scalar(a + 3, 1, 0x11);
    mem.store_scalar(a + 3 * 65536 + 8, 2, 0x2222);
    mem.spec_block(1);
    mem.store_scalar(a + 2 * 65536 + 2000, 8, 0x3333333333333333ull);   // its own: written, then undone with the rest
    VCHECK(stops([&] { mem.store_scalar(a + 65536, 1, 9); }));           // block 0's
    VCHECK(stops([&] { mem.load_scalar(a + 20000, 4); }));               // (and every access after the stop)
    const MemoryManager::SpecOutcome out = t.end();
    VCHECK(!out.committed);
    VCHECK(std::string(out.why) == "two blocks wrote the same bytes");
    VCHECK(all() == init);
  }
  {
    // The shadow is empty again: what was block 0's in the last trial is anyone's now.
    Trial t(mem);
    mem.spec_block(1);
    mem.store_scalar(a + 65536, 4, 0x01020304);
    mem.spec_block(0);
    mem.read(a + 3, &v, 1);
    VCHECK(t.end().committed);
    init[65536] = 4, init[65537] = 3, init[65538] = 2, init[65539] = 1;
    VCHECK(all() == init);
  }
  {
    // A block may read and write its own words in any order; a second reader of a word a block then writes stops that write.
    Trial t(mem);
    mem.load_scalar(a + 200, 4);
    mem.store_scalar(a + 200, 4, 1);
    mem.load_scalar(a + 200, 4);
    mem.load_scalar(a + 300, 4);
    mem.spec_block(1);
    mem.load_scalar(a + 300, 4);
    mem.spec_block(0);
    VCHECK(stops([&] { mem.store_scalar(a + 300, 4, 2); }));
    const MemoryManager::SpecOutcome out = t.end();
    VCHECK(std::string(out.why) == "a block wrote what another block read");
    VCHECK(all() == init);
  }
}

VTEST(a_device_with_memory_the_shadow_cannot_see_is_not_tried) {
  MemoryManager plain{1 << 24};
  VCHECK(!plain.spec_possible());   // nothing allocated yet: nothing to shadow
  plain.alloc(64);
  VCHECK(plain.spec_possible());
  MemoryManager vmm{1 << 26};
  vmm.alloc(64);
  const uint64_t va = vmm.reserve(MemoryManager::kVmmGranularity, MemoryManager::kVmmGranularity);
  VCHECK(!vmm.spec_possible());   // a reservation: the same bytes may be mapped at two addresses
  vmm.address_free(va, MemoryManager::kVmmGranularity);
  VCHECK(vmm.spec_possible());
}

VTEST_MAIN
