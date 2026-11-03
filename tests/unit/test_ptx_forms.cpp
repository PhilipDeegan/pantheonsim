// PTX forms that real producers emit (drjit-core, JIT compilers) and that the frontend used to refuse:
//   * a negated predicate source operand (`and.pred %p2, !%p0, %p1;`)    issue 474
//   * `ldu` (uniform load), which reads what `ld` reads    issue 472
// Each result is compared with what the PTX ISA says the instruction computes, worked out here on the host.
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/runtime/runtime.hpp"
#include "vtest.hpp"

using namespace vgpu;
using vgpu::exec::LaunchConfig;

namespace {
const char* kHeader = ".version 8.3\n.target sm_86\n.address_size 64\n";

std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}

// Loads `ptx` through the runtime (which places the module's globals and its function addresses), launches kernel `k` with
// `threads` threads and one pointer argument to `words` zeroed 32-bit words, and returns the words.
std::vector<uint32_t> run(const std::string& ptx, uint32_t threads, size_t words) {
  runtime::Runtime rt(load_gpu("nvidia/a10"));
  auto& dev = rt.device(0);
  const uint64_t mod = dev.load_module(ptx);
  const ptx::EntryFn* fn = dev.get_function(mod, "k");
  const uint64_t out = dev.memory().alloc(words * 4);
  for (size_t i = 0; i < words; ++i) dev.memory().store_scalar(out + 4 * i, 4, 0);
  LaunchConfig cfg;
  cfg.block = {threads, 1, 1};
  dev.launch(*fn, cfg, {arg_u64(out)}, dev.symbols(mod));
  std::vector<uint32_t> r(words);
  for (size_t i = 0; i < words; ++i) r[i] = static_cast<uint32_t>(dev.memory().load_scalar(out + 4 * i, 4));
  return r;
}
}  // namespace

// ---- 474: a negated predicate source ----

// Every thread holds a different pair of predicate values (lane bit 0 and bit 1), so all four combinations occur in a warp.
// Each operation writes 1 or 0 through selp. The expected value is the boolean formula, written on the host.
VTEST(a_negated_predicate_source_in_and_or_xor_not_and_selp) {
  const std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 po)
{
  .reg .pred %p<16>;
  .reg .b32 %r<16>;
  .reg .b64 %rd<4>;
  ld.param.u64 %rd0, [po];
  cvta.to.global.u64 %rd0, %rd0;
  mov.u32 %r0, %tid.x;
  and.b32 %r1, %r0, 1;
  and.b32 %r2, %r0, 2;
  setp.ne.u32 %p0, %r1, 0;      // a = lane bit 0
  setp.ne.u32 %p1, %r2, 0;      // b = lane bit 1
  and.pred %p2, !%p0, %p1;
  xor.pred %p3, %p0, !%p1;
  or.pred  %p4, !%p0, !%p1;
  and.pred %p5, %p0, !%p1;
  xor.pred %p6, !%p0, !%p1;
  not.pred %p7, !%p0;
  selp.u32 %r3, 1, 0, !%p0;
  and.pred %p8, !%p0, 1;
  setp.eq.and.u32 %p9, %r1, 0, !%p1;
  selp.u32 %r4, 1, 0, %p2;
  selp.u32 %r5, 1, 0, %p3;
  selp.u32 %r6, 1, 0, %p4;
  selp.u32 %r7, 1, 0, %p5;
  selp.u32 %r8, 1, 0, %p6;
  selp.u32 %r9, 1, 0, %p7;
  selp.u32 %r10, 1, 0, %p8;
  selp.u32 %r11, 1, 0, %p9;
  mul.wide.u32 %rd1, %r0, 36;
  add.u64 %rd2, %rd0, %rd1;
  st.global.u32 [%rd2+0], %r4;
  st.global.u32 [%rd2+4], %r5;
  st.global.u32 [%rd2+8], %r6;
  st.global.u32 [%rd2+12], %r7;
  st.global.u32 [%rd2+16], %r8;
  st.global.u32 [%rd2+20], %r9;
  st.global.u32 [%rd2+24], %r3;
  st.global.u32 [%rd2+28], %r10;
  st.global.u32 [%rd2+32], %r11;
  ret;
}
)";
  const std::vector<uint32_t> r = run(ptx, 32, 32 * 9);
  for (uint32_t t = 0; t < 32; ++t) {
    const bool a = t & 1, b = (t >> 1) & 1;
    const uint32_t want[9] = {!a && b, a != !b, !a || !b, a && !b, !a != !b, a, !a, !a, !a && !b};
    for (int i = 0; i < 9; ++i) VCHECK_EQ(uint64_t{r[t * 9 + i]}, uint64_t{want[i]});
  }
}

// The guard form `@!%p` keeps working next to the new operand form.
VTEST(a_negated_guard_still_works_beside_a_negated_source) {
  const std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 po)
{
  .reg .pred %p<4>;
  .reg .b32 %r<4>;
  .reg .b64 %rd<4>;
  ld.param.u64 %rd0, [po];
  cvta.to.global.u64 %rd0, %rd0;
  mov.u32 %r0, %tid.x;
  and.b32 %r1, %r0, 1;
  setp.ne.u32 %p0, %r1, 0;
  mov.u32 %r2, 5;
  @!%p0 mov.u32 %r2, 9;
  and.pred %p1, !%p0, !%p0;
  @!%p1 add.u32 %r2, %r2, 100;
  mul.wide.u32 %rd1, %r0, 4;
  add.u64 %rd2, %rd0, %rd1;
  st.global.u32 [%rd2], %r2;
  ret;
}
)";
  const std::vector<uint32_t> r = run(ptx, 32, 32);
  for (uint32_t t = 0; t < 32; ++t) VCHECK_EQ(uint64_t{r[t]}, uint64_t{(t & 1) ? 105u : 9u});
}

// ---- 472: ldu ----

VTEST(ldu_loads_what_ld_loads_scalar_and_vector) {
  const std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 po)
{
  .reg .b64 %rd<3>;
  .reg .b32 %r<8>;
  ld.param.u64 %rd1, [po];
  cvta.to.global.u64 %rd1, %rd1;
  mov.u32 %r0, 41;
  st.global.u32 [%rd1+16], %r0;
  mov.u32 %r0, 1;
  st.global.u32 [%rd1+32], %r0;
  mov.u32 %r0, 2;
  st.global.u32 [%rd1+36], %r0;
  mov.u32 %r0, 3;
  st.global.u32 [%rd1+40], %r0;
  mov.u32 %r0, 4;
  st.global.u32 [%rd1+44], %r0;
  ldu.global.u32 %r1, [%rd1+16];
  add.u32 %r1, %r1, 1;
  st.global.u32 [%rd1], %r1;
  ldu.global.v4.u32 {%r2, %r3, %r4, %r5}, [%rd1+32];
  add.u32 %r6, %r2, %r3;
  add.u32 %r6, %r6, %r4;
  add.u32 %r6, %r6, %r5;
  st.global.u32 [%rd1+4], %r6;
  ret;
}
)";
  const std::vector<uint32_t> r = run(ptx, 1, 12);
  VCHECK_EQ(uint64_t{r[0]}, uint64_t{42});
  VCHECK_EQ(uint64_t{r[1]}, uint64_t{1 + 2 + 3 + 4});
}

VTEST(ldu_outside_global_is_refused) {
  const std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 po)
{
  .reg .b32 %r<2>;
  .reg .b64 %rd<2>;
  ldu.shared.u32 %r1, [%rd1];
  ret;
}
)";
  auto e = VCAPTURE(Error, ptx::parse(ptx));
  VCHECK_CONTAINS(std::string(e.what()), "ldu outside .global");
}

VTEST_MAIN
