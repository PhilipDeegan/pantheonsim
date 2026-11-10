// The part of the JIT's generated code that does not change from one region to the next. It is embedded in the simulator (CMake
// writes it, and src/simd/kernels.inc, into jit_embed.cpp), written into the cache directory at first use, verified by its hash, and
// its hash is part of every key: the generated code needs nothing from the source tree at run time.
//
// The lane kernels are src/simd/kernels.inc itself, the text the simulator's own kernels (src/simd/simd.cpp) are compiled from,
// included here with VGPU_LANES_ONLY so that it leaves out what the SASS ALU forms do not use. One source for the arithmetic: a form's
// generated code calls the kernel the tier's handler calls (the generator's include of this file defines VGPU_SIMD_AVX2 first: 1 for the
// kernels of the AVX2 level, 0 for the baseline, as the simulator is running).
#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#if defined(__x86_64__)
#include <emmintrin.h>
#include <immintrin.h>
#endif

#define VGPU_LANES_ONLY 1
// The generated code inlines every kernel it calls: the arithmetic of a region is then one function the compiler can schedule as a whole.
#define VGPU_KN [[gnu::always_inline]] inline
#define VGPU_KA [[gnu::always_inline]]

namespace {

namespace jk {
#define VGPU_SIMD_NS jk_kernels
#include "vgpu_jit_kernels.inc"
#undef VGPU_SIMD_NS
}  // namespace jk

using u32 = uint32_t;
using u64 = uint64_t;
using namespace jk::jk_kernels;

// Modifiers on a source that has to be worked out when the instruction runs (tier_mod): for floats |x| then -x on the bits; for
// integers -x as 0 - x, then ~x.
[[gnu::always_inline]] inline u32 tmod(bool flt, unsigned mod, u32 v) {
  if (flt) {
    if (mod & 4u) v &= 0x7fffffffu;
    if (mod & 1u) v ^= 0x80000000u;
  } else {
    if (mod & 1u) v = 0u - v;
    if (mod & 2u) v = ~v;
  }
  return v;
}

[[gnu::always_inline]] inline void splat(u32* t, u32 v) {
  const V8u x = {v, v, v, v, v, v, v, v};
  for (unsigned i = 0; i < 32; i += 8) std::memcpy(t + i, &x, sizeof x);
}

// A register with a modifier: its 32 lanes worked through tmod.
[[gnu::always_inline]] inline const u32* dynreg(u32* t, const u32* r, bool flt, unsigned mod) {
  for (unsigned l = 0; l < 32; ++l) t[l] = tmod(flt, mod, r[l]);
  return t;
}

// A result row to the destination register for the executing lanes (tier_store): the whole row, or a blend of the lanes.
[[gnu::always_inline]] inline void store_row(u32* d, u32 ex, const u32* r) {
  if (ex == 0xffffffffu) {
    for (unsigned i = 0; i < 32; i += 8) {
      V8u v;
      std::memcpy(&v, r + i, sizeof v);
      std::memcpy(d + i, &v, sizeof v);
    }
  } else {
    blend_m_k(d, r, ex);
  }
}

// The register file's row and a local copy of it (register residence).
[[gnu::always_inline]] inline void ldrow(u32* x, const u32* r) {
  for (unsigned i = 0; i < 32; i += 8) {
    V8u v;
    std::memcpy(&v, r + i, sizeof v);
    std::memcpy(x + i, &v, sizeof v);
  }
}
[[gnu::always_inline]] inline void strow(u32* r, const u32* x) {
  for (unsigned i = 0; i < 32; i += 8) {
    V8u v;
    std::memcpy(&v, x + i, sizeof v);
    std::memcpy(r + i, &v, sizeof v);
  }
}

// Fused runs (round sp48): a run of register-only ops is computed block by block of eight lanes, every op of the run on one block before the
// next block, so that a value a later op of the run reads is in a host register and not in a 128-byte row in memory. These are the kernels of
// src/simd/kernels.inc for one block (iadd3_k, imad_k and lop3_k are written out by the generator; the float ones are here because of the NaN
// rule): the same expression, on a vector instead of a row. Only for the AVX2 level (a 32-byte vector argument is not a baseline x86-64 one).
#if VGPU_SIMD_AVX2
[[gnu::always_inline]] inline V8u ld8(const u32* p) {
  V8u v;
  std::memcpy(&v, p, sizeof v);
  return v;
}
[[gnu::always_inline]] inline void st8(u32* p, V8u v) { std::memcpy(p, &v, sizeof v); }
[[gnu::always_inline]] inline V8u sp8(u32 v) { return V8u{v, v, v, v, v, v, v, v}; }
// The lanes q .. q + 7 of a lane mask as a vector of all-ones or zero words (store_masked's `sel`).
[[gnu::always_inline]] inline V8u lm8(u32 ex, unsigned q) {
  const V8u pick = {1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u};
  return reinterpret_cast<V8u>((pick & sp8((ex >> q) & 0xffu)) != 0);
}
[[gnu::always_inline]] inline V8u bl8(V8u m, V8u r, V8u old) { return (old & ~m) | (r & m); }
[[gnu::always_inline]] inline V8u tmod8(bool flt, unsigned mod, V8u v) {   // tmod, lane by lane
  if (flt) {
    if (mod & 4u) v &= sp8(0x7fffffffu);
    if (mod & 1u) v ^= sp8(0x80000000u);
  } else {
    if (mod & 1u) v = sp8(0u) - v;
    if (mod & 2u) v = ~v;
  }
  return v;
}
[[gnu::always_inline]] inline V8u fadd8(V8u a, V8u b) { return VGPU_CANON_NAN(reinterpret_cast<V8f>(a) + reinterpret_cast<V8f>(b)); }
[[gnu::always_inline]] inline V8u fmul8(V8u a, V8u b) { return VGPU_CANON_NAN(reinterpret_cast<V8f>(a) * reinterpret_cast<V8f>(b)); }
[[gnu::always_inline]] inline V8u ffma8(V8u a, V8u b, V8u c) {
  return VGPU_CANON_NAN(reinterpret_cast<V8f>(_mm256_fmadd_ps(reinterpret_cast<__m256>(a), reinterpret_cast<__m256>(b), reinterpret_cast<__m256>(c))));
}
#endif

// The scalar comparisons of the uniform datapath (exec_ops.inc's int_cmp and bool_op).
[[gnu::always_inline]] inline bool icmp1(unsigned cmp, bool sgn, u32 a, u32 b) {
  const bool lt = sgn ? static_cast<int32_t>(a) < static_cast<int32_t>(b) : a < b;
  const bool gt = sgn ? static_cast<int32_t>(a) > static_cast<int32_t>(b) : a > b;
  switch (cmp) {
    case 0: return false;
    case 1: return lt;
    case 2: return a == b;
    case 3: return !gt;
    case 4: return gt;
    case 5: return a != b;
    case 6: return !lt;
    default: return true;
  }
}
[[gnu::always_inline]] inline bool bool_op1(unsigned op, bool a, bool b) { return op == 0 ? (a && b) : op == 1 ? (a || b) : (a != b); }

inline u32 combine(unsigned bop, u32 r, u32 acc) { return bop == 0 ? (r & acc) : bop == 1 ? (r | acc) : (r ^ acc); }

alignas(32) const u32 kZeros32[32] = {};
alignas(32) const u32 kOnes32[32] = {~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u,
                                      ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u};

}  // namespace
