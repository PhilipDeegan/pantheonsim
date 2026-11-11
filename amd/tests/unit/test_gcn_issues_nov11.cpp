// Three bugs of the simulated MI250X and MI300X that other people's programs ran into (issues #578, #579 and #580), each as a small
// kernel of amd/tests/data/asm_issues_nov11.s. The expected values are written from the ISA guides, not taken from the simulator.
//   #578  a scratch access whose SGPR offset is negative and whose instruction offset brings it back (CDNA3 ISA 10.5: "flat_scratch.addr +
//         swizzle(sgpr_offset + inst.offset, threadID)") reached for 2^32 instead of the work-item's own address.
//   #579  "s_load_dwordx2 vcc, s[94:95], 0x0": the loaded pair went to s0 and s1 (the private segment buffer on gfx90a), not to VCC.
//   #580  "v_pk_add_f32 v[2:3], v[2:3], 0 neg_lo:[1,1] neg_hi:[1,1]": the constant 0 is zero in both halves of the packed operand.
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vgpu/error.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

amd::CodeObject object(const char* target) {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/asm_issues_nov11." + target + ".o";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no code object at " + path);
  return amd::load_code_object(std::string((std::istreambuf_iterator<char>(in)), {}), path);
}

template <typename T>
uint64_t put(MemoryManager& mem, const std::vector<T>& v) {
  const uint64_t p = mem.alloc(v.size() * sizeof(T));
  mem.write(p, v.data(), v.size() * sizeof(T));
  return p;
}

template <typename T>
std::vector<T> get(MemoryManager& mem, uint64_t p, size_t n) {
  std::vector<T> out(n);
  mem.read(p, out.data(), n * sizeof(T));
  return out;
}

void launch(const amd::CodeObject& o, const char* name, MemoryManager& mem, const std::vector<uint64_t>& args) {
  const amd::Kernel* k = amd::find_kernel(o, name);
  if (!k) throw vtest::Failure(std::string("no kernel named ") + name);
  std::vector<uint8_t> buf(k->kernarg_size, 0);
  for (size_t i = 0; i < args.size(); ++i) std::memcpy(&buf[8 * i], &args[i], 8);
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = mem.alloc(buf.size());
  mem.write(d.kernarg, buf.data(), buf.size());
  d.group_size[0] = 64;
  amd::execute(d, mem);
}

uint32_t bits(float f) { return std::bit_cast<uint32_t>(f); }

}  // namespace

// #580: the packed float instructions with the constant 0 as a source. The add is -x + -0, the multiply x * 0 and the fused multiply-add
// x * x + 0, in each half.
VTEST(a_packed_float_instruction_reads_the_constant_zero_in_both_halves) {
  for (const char* target : {"gfx90a", "gfx942"}) {
    const amd::CodeObject o = object(target);
    MemoryManager mem(16ull << 20);
    std::vector<float> in(128);
    for (int l = 0; l < 64; ++l) {
      in[2 * l] = l % 7 == 0 ? 0.0f : (l % 2 ? 1.0f : -1.0f) * (0.25f + 0.5f * l);
      in[2 * l + 1] = l % 11 == 0 ? -0.0f : (l % 3 ? -1.0f : 1.0f) * (3.0f - 0.125f * l);
    }
    const uint64_t pin = put(mem, in), pout = mem.alloc(3 * 512);
    launch(o, "pk_zero", mem, {pin, pout});
    const std::vector<uint32_t> got = get<uint32_t>(mem, pout, 3 * 128);
    int wrong = 0;
    for (int i = 0; i < 128; ++i) {
      const float x = in[i];
      wrong += got[i] != bits(-x + -0.0f);
      wrong += got[128 + i] != bits(x * 0.0f);
      wrong += got[256 + i] != bits(std::fmaf(x, x, 0.0f));
    }
    if (wrong) std::fprintf(stderr, "%s: %d wrong\n", target, wrong);
    VCHECK_EQ(wrong, 0);
  }
}

// What stays refused: a nonzero constant read as the second half of a packed operand (the guides give the expansion of a 64-bit constant for
// literals, 6.2.1.1, and not what a packed instruction's inline float constant means in its second half).
VTEST(the_second_half_of_a_nonzero_packed_constant_is_still_refused) {
  const amd::CodeObject o = object("gfx942");
  MemoryManager mem(16ull << 20);
  const uint64_t pin = put(mem, std::vector<float>(128, 2.0f)), pout = mem.alloc(512);
  bool refused = false;
  try {
    launch(o, "pk_const", mem, {pin, pout});
  } catch (const Error& e) {
    refused = std::string(e.what()).find("second part of a constant") != std::string::npos;
  }
  VCHECK(refused);
}

// #579: a scalar load whose destination is VCC (or half of it) fills VCC and leaves s0 and s1 alone.
VTEST(a_scalar_load_into_vcc_does_not_write_s0_and_s1) {
  for (const char* target : {"gfx90a", "gfx942"}) {
    const amd::CodeObject o = object(target);
    MemoryManager mem(16ull << 20);
    const std::vector<uint32_t> in = {0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u};
    const uint64_t pin = put(mem, in), pout = mem.alloc(32);
    launch(o, "smem_vcc", mem, {pin, pout});
    const std::vector<uint32_t> got = get<uint32_t>(mem, pout, 6);
    VCHECK_EQ(got[0], 0x11111111u);   // vcc_lo, vcc_hi after the dwordx2 load
    VCHECK_EQ(got[1], 0x22222222u);
    VCHECK_EQ(got[2], 1u);            // s[0:1] still the kernarg pointer
    VCHECK_EQ(got[3], 0x22222222u);   // vcc_hi after loading only vcc_lo
    VCHECK_EQ(got[4], 0x33333333u);   // vcc_lo
    VCHECK_EQ(got[5], 0x44444444u);   // vcc_hi after loading it
  }
}

// #578: scratch at s6 = -32 with instruction offset 32 is the work-item's byte 0, and at s7 = -16 with 48 its byte 32.
VTEST(a_scratch_access_sums_a_negative_scalar_offset_and_the_instruction_offset_in_32_bits) {
  const amd::CodeObject o = object("gfx942");
  MemoryManager mem(16ull << 20);
  const uint64_t pout = mem.alloc(256 + 64 * 16);
  launch(o, "neg_scratch", mem, {pout});
  const std::vector<uint32_t> got = get<uint32_t>(mem, pout, 64 + 256);
  int wrong = 0;
  for (uint32_t l = 0; l < 64; ++l) {
    wrong += got[l] != 100 + l;
    for (uint32_t k = 0; k < 4; ++k) wrong += got[64 + 4 * l + k] != 1000 * (k + 1) + l;
  }
  if (wrong) std::fprintf(stderr, "%d wrong\n", wrong);
  VCHECK_EQ(wrong, 0);
}

VTEST_MAIN
