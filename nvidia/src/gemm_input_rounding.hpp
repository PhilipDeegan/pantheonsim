// The reduced-precision inputs of cuBLAS's tensor-core GEMM paths.
//
// cublasSetMathMode(CUBLAS_TF32_TENSOR_OP_MATH), CUBLAS_COMPUTE_32F_FAST_TF32 (also on cublasLt) and the
// FAST_16F / FAST_16BF compute types make a tensor-core kernel convert every FP32 input element to a narrower
// format before the multiply; the products are exact and the sums are FP32. The host-computed GEMMs here do
// the same conversion to A and B (never to C) and then run their usual FP32 sum.
//
// Measured on an RTX 3060 (GA106, cuBLAS 13.0), c = a * 1 with probe values around every rounding point:
//   * TF32 is round to nearest with ties AWAY from zero (cvt.rna.tf32.f32), 10 stored mantissa bits: 1 + 2^-11
//     becomes 1 + 2^-10 (round-to-even would give 1, truncation 1). Infinities and NaNs pass, and a value
//     within half a TF32 step of FLT_MAX rounds to infinity; subnormals are not flushed.
//   * Random 256^3 products against a double-precision reference on rounded inputs: error 4e-5 with the TF32
//     RNA model, 2e-3 with round-to-even, 7e-2 with truncation, 2.7e-2 unrounded.
//   * Which routines: Sgemm, SgemmBatched, SgemmStridedBatched, SgemmEx(R_32F), Cgemm (real and imaginary
//     parts), GemmEx and its batched forms take TF32 from the handle's math mode (value & 3 == 3,
//     CUBLAS_TF32_TENSOR_OP_MATH) or from compute type FAST_TF32 (whatever the math mode, PEDANTIC included).
//     GemmEx with 32F_PEDANTIC and the default mode do not round. cublasLtMatmul never reads the handle's
//     mode: only compute type FAST_TF32 rounds. Ssyrk, Ssymm and Strmm do not round in TF32 mode.
//   * The deprecated CUBLAS_TENSOR_OP_MATH (value & 3 == 1) rounds Sgemm and GemmEx(32F) to FP16 (round to
//     nearest even); FAST_16F does the same and FAST_16BF rounds to BF16 (nearest even), on any mode.
//   * NVIDIA_TF32_OVERRIDE=0 in the environment turns TF32 off everywhere, explicit FAST_TF32 included.
//   * The card only takes the tensor-core kernel from a size on: for small products (every probe below
//     m*n*k = 2^20, and any m < 4 or n < 2 -- gemv-like shapes) the answer is the plain FP32 one. Measured
//     with a 1 + 2^-11 probe over m, n, k in {1 ... 1024} (NN, NT, TN, TT, Sgemm, GemmEx and Lt gave the same
//     boundary): rounding starts at m*n*k >= 2^21 with m >= 4 and n >= 2, give or take cuBLAS's own heuristic
//     (for m, n >= 256 with a small k, and a few (m, n) pairs, the card keeps the FP32 kernel up to 2^22-2^24).
//     That is the rule used here; it follows the card's choice of kernel for the sizes real programs use
//     and is the one thing this file does not take from a measurement of every size.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace vgpu::gemmround {

enum class Kind { None, Tf32, Fp16, Bf16 };

inline float from_bits(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
inline uint32_t to_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}

// Round to a 10-bit mantissa, ties away from zero; inf and NaN are kept.
inline float to_tf32(float x) {
  uint32_t u = to_bits(x);
  if ((u & 0x7f800000u) == 0x7f800000u) return x;
  u += 0x1000u;       // half a step: carries into the kept bits exactly when the dropped bits are >= half
  return from_bits(u & ~0x1fffu);
}

// Round to a 7-bit mantissa, nearest even; NaN is kept.
inline float to_bf16(float x) {
  uint32_t u = to_bits(x);
  if ((u & 0x7f800000u) == 0x7f800000u) return x;
  u += 0x7fffu + ((u >> 16) & 1u);
  return from_bits(u & 0xffff0000u);
}

// Round to the nearest IEEE half (ties to even, overflow to infinity, subnormal halves kept) and widen back.
inline float to_fp16(float x) {
  const uint32_t u = to_bits(x);
  if ((u & 0x7f800000u) == 0x7f800000u) return x;
  const uint32_t sign = u & 0x80000000u;
  const float a = from_bits(u & 0x7fffffffu);
  if (a >= 65520.0f) return from_bits(sign | 0x7f800000u);
  float r;
  if (a < 6.103515625e-05f) {   // below the smallest normal half: a multiple of 2^-24
    r = std::nearbyint(a * 16777216.0f) / 16777216.0f;
  } else {
    uint32_t v = to_bits(a);
    v += 0xfffu + ((v >> 13) & 1u);
    r = from_bits(v & ~0x1fffu);
  }
  return from_bits(sign | to_bits(r));
}

inline float round_to(float x, Kind k) {
  switch (k) {
    case Kind::Tf32: return to_tf32(x);
    case Kind::Fp16: return to_fp16(x);
    case Kind::Bf16: return to_bf16(x);
    default: return x;
  }
}

inline void round_in_place(float* p, size_t n, Kind k) {
  if (k == Kind::None) return;
  for (size_t i = 0; i < n; ++i) p[i] = round_to(p[i], k);
}

// NVIDIA_TF32_OVERRIDE=0 disables TF32 for the process. Read on first use.
inline bool tf32_overridden_off() {
  static const bool off = [] {
    const char* e = std::getenv("NVIDIA_TF32_OVERRIDE");
    return e && e[0] == '0' && e[1] == 0;
  }();
  return off;
}

// Whether the card's heuristic picks a tensor-core kernel for an m x n x k product (see the top of this file).
inline bool tensor_core_shape(int64_t m, int64_t n, int64_t k) {
  return m >= 4 && n >= 2 && k >= 1 && m * n * k >= (int64_t(1) << 21);
}

// The rounding in force for one product, given the format its compute type asks for (Kind::None with
// `explicit_kind` false: the handle's math mode decides).
inline Kind effective(Kind requested, bool explicit_kind, bool pedantic_compute, int math_mode, int64_t m, int64_t n,
                      int64_t k) {
  Kind kind = Kind::None;
  if (explicit_kind) kind = requested;
  else if (!pedantic_compute) kind = (math_mode & 3) == 3 ? Kind::Tf32 : (math_mode & 3) == 1 ? Kind::Fp16 : Kind::None;
  if (kind == Kind::Tf32 && tf32_overridden_off()) kind = Kind::None;
  if (kind != Kind::None && !tensor_core_shape(m, n, k)) kind = Kind::None;
  return kind;
}

}  // namespace vgpu::gemmround
