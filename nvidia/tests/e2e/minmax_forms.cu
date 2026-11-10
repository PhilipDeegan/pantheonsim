// min / max in every form PTX has, against a model of the PTX ISA (pantheonsim issue #547 and the siblings of its form).
//
// {min,max}{.ftz}{.NaN}{.xorsign.abs}.{f32,f16,f16x2,bf16,bf16x2}, the plain f64 and integer forms, the sm_90 {min,max}{.relu}.{s32,s16x2},
// and the three-input f32 {min,max}{.ftz}{.NaN}{.abs}.f32 of sm_100, each run over a spread of operands (zeros of both signs, infinities,
// quiet NaNs with payloads, subnormals, the largest finite values, ties of magnitude) and compared with a model written from the ISA:
//   a NaN input loses to a number (both NaN: a NaN), .NaN returns a NaN when either input is one, min(+0, -0) = -0 and max(+0, -0) = +0,
//   .ftz flushes subnormal inputs to zero of their sign, .xorsign.abs takes the smaller (larger) magnitude with the XOR of the signs,
//   .relu clamps the result at 0.
// Every form here also gives these answers on an RTX 3060 (the ones its architecture, sm_86, has; this program passes there) -- bit for bit
// except which NaN is returned, which is not compared. Run by e2e_minmax_forms on every architecture from Ampere on, on the SASS and on the PTX.
//
// Prints PASS on the last line.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

static int g_fail = 0, g_run = 0, g_skip = 0;

// ---- the model ----------------------------------------------------------------------------------------------
struct Fmt { int bits, mant, exp; };
static const Fmt F32{32, 23, 8}, F16{16, 10, 5}, BF16{16, 7, 8};
static bool is_nan(uint32_t v, Fmt f) { const uint32_t e = ((1u << f.exp) - 1) << f.mant, m = (1u << f.mant) - 1; return (v & e) == e && (v & m); }
static bool is_zero(uint32_t v, Fmt f) { return (v & ~(1u << (f.bits - 1)) & (f.bits == 32 ? 0xffffffffu : 0xffffu)) == 0; }
static bool is_sub(uint32_t v, Fmt f) { return (v & (((1u << f.exp) - 1) << f.mant)) == 0 && (v & ((1u << f.mant) - 1)); }
static uint32_t sign_bit(Fmt f) { return 1u << (f.bits - 1); }
static double value(uint32_t v, Fmt f) {
  const bool neg = v & sign_bit(f);
  const int e = (v >> f.mant) & ((1 << f.exp) - 1);
  const double m = v & ((1u << f.mant) - 1);
  const int bias = (1 << (f.exp - 1)) - 1;
  double r;
  if (e == (1 << f.exp) - 1) r = m ? NAN : INFINITY;
  else if (e == 0) r = std::ldexp(m, 1 - bias - f.mant);
  else r = std::ldexp(1.0 + m / (1 << f.mant), e - bias);
  return neg ? -r : r;
}
struct Mods { bool ftz = false, nan = false, xs = false, abs3 = false; };
// A result: a bit pattern, or any NaN.
struct Res { bool any_nan; uint32_t bits; };
static Res model_min_max(bool is_min, uint32_t a, uint32_t b, Fmt f, const Mods& m) {
  if (m.ftz && f.bits != 0) {   // (bf16 has no .ftz)
    if (is_sub(a, f)) a &= sign_bit(f);
    if (is_sub(b, f)) b &= sign_bit(f);
  }
  const uint32_t sgn = (a ^ b) & sign_bit(f);
  if (m.xs) { a &= ~sign_bit(f); b &= ~sign_bit(f); }
  const bool na = is_nan(a, f), nb = is_nan(b, f);
  if ((m.nan && (na || nb)) || (na && nb)) return {true, 0};
  uint32_t r;
  if (na) r = b;
  else if (nb) r = a;
  else if (is_zero(a, f) && is_zero(b, f)) {
    const bool an = a & sign_bit(f), bn = b & sign_bit(f);
    r = is_min ? (an ? a : b) : (an ? b : a);
  } else {
    const double x = value(a, f), y = value(b, f);
    r = is_min ? (x < y ? a : b) : (x > y ? a : b);
  }
  if (m.xs) r = (r & ~sign_bit(f)) | sgn;
  return {false, r};
}
static Res model_min_max3(bool is_min, uint32_t a, uint32_t b, uint32_t c, const Mods& m) {
  const Fmt f = F32;
  uint32_t v[3] = {a, b, c};
  bool any = false;
  for (uint32_t& x : v) {
    if (m.ftz && is_sub(x, f)) x &= sign_bit(f);
    if (m.abs3) x &= ~sign_bit(f);
    any = any || is_nan(x, f);
  }
  if (m.nan && any) return {true, 0};
  Mods plain;
  Res r{false, v[0]};
  bool r_nan = is_nan(v[0], f);
  for (int i = 1; i < 3; ++i) {
    const bool x_nan = is_nan(v[i], f);
    if (r_nan && x_nan) { r = {true, 0}; continue; }
    if (r_nan) { r = {false, v[i]}; r_nan = false; continue; }
    if (x_nan) continue;
    r = model_min_max(is_min, r.bits, v[i], f, plain);
  }
  return r;
}

// ---- operands -----------------------------------------------------------------------------------------------
static std::vector<uint32_t> vals(Fmt f) {
  std::vector<uint32_t> v;
  const uint32_t s = sign_bit(f);
  const uint32_t one = f.bits == 32 ? 0x3f800000u : f.mant == 10 ? 0x3c00u : 0x3f80u;
  const uint32_t inf = ((1u << f.exp) - 1) << f.mant, qnan = inf | (1u << (f.mant - 1)), snan = inf | 1u | (1u << (f.mant - 2));
  const uint32_t maxf = inf - 1, minn = 1u << f.mant, sub = 1u, sub2 = (1u << f.mant) - 1;
  const uint32_t two = f.bits == 32 ? 0x40000000u : f.mant == 10 ? 0x4000u : 0x4000u;
  for (uint32_t x : {0u, one, two, maxf, minn, sub, sub2, inf, qnan, snan, one + 3u})
    { v.push_back(x); v.push_back(x | s); }
  return v;
}

// ---- device ---------------------------------------------------------------------------------------------------
__global__ void k_min_f32(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  float r; asm volatile("min.f32 %0, %1, %2;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
__global__ void k_min_xorsign_abs_f32(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  float r; asm volatile("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
__global__ void k_min_NaN_f32(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  float r; asm volatile("min.NaN.f32 %0, %1, %2;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
__global__ void k_min_NaN_xorsign_abs_f32(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  float r; asm volatile("min.NaN.xorsign.abs.f32 %0, %1, %2;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
__global__ void k_min_ftz_f32(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  float r; asm volatile("min.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
__global__ void k_min_ftz_xorsign_abs_f32(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  float r; asm volatile("min.ftz.xorsign.abs.f32 %0, %1, %2;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
__global__ void k_min_ftz_NaN_f32(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  float r; asm volatile("min.ftz.NaN.f32 %0, %1, %2;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
__global__ void k_min_ftz_NaN_xorsign_abs_f32(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  float r; asm volatile("min.ftz.NaN.xorsign.abs.f32 %0, %1, %2;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
__global__ void k_max_f32(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  float r; asm volatile("max.f32 %0, %1, %2;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
__global__ void k_max_xorsign_abs_f32(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  float r; asm volatile("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
__global__ void k_max_NaN_f32(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  float r; asm volatile("max.NaN.f32 %0, %1, %2;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
__global__ void k_max_NaN_xorsign_abs_f32(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  float r; asm volatile("max.NaN.xorsign.abs.f32 %0, %1, %2;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
__global__ void k_max_ftz_f32(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  float r; asm volatile("max.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
__global__ void k_max_ftz_xorsign_abs_f32(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  float r; asm volatile("max.ftz.xorsign.abs.f32 %0, %1, %2;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
__global__ void k_max_ftz_NaN_f32(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  float r; asm volatile("max.ftz.NaN.f32 %0, %1, %2;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
__global__ void k_max_ftz_NaN_xorsign_abs_f32(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  float r; asm volatile("max.ftz.NaN.xorsign.abs.f32 %0, %1, %2;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
__global__ void k_min_f16(const unsigned short* a, const unsigned short* b, unsigned short* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned short r; asm volatile("min.f16 %0, %1, %2;" : "=h"(r) : "h"(a[i]), "h"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_min_xorsign_abs_f16(const unsigned short* a, const unsigned short* b, unsigned short* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  unsigned short r; asm volatile("min.xorsign.abs.f16 %0, %1, %2;" : "=h"(r) : "h"(a[i]), "h"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_min_NaN_f16(const unsigned short* a, const unsigned short* b, unsigned short* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned short r; asm volatile("min.NaN.f16 %0, %1, %2;" : "=h"(r) : "h"(a[i]), "h"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_min_NaN_xorsign_abs_f16(const unsigned short* a, const unsigned short* b, unsigned short* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  unsigned short r; asm volatile("min.NaN.xorsign.abs.f16 %0, %1, %2;" : "=h"(r) : "h"(a[i]), "h"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_min_ftz_f16(const unsigned short* a, const unsigned short* b, unsigned short* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned short r; asm volatile("min.ftz.f16 %0, %1, %2;" : "=h"(r) : "h"(a[i]), "h"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_min_ftz_xorsign_abs_f16(const unsigned short* a, const unsigned short* b, unsigned short* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  unsigned short r; asm volatile("min.ftz.xorsign.abs.f16 %0, %1, %2;" : "=h"(r) : "h"(a[i]), "h"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_min_ftz_NaN_f16(const unsigned short* a, const unsigned short* b, unsigned short* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned short r; asm volatile("min.ftz.NaN.f16 %0, %1, %2;" : "=h"(r) : "h"(a[i]), "h"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_min_ftz_NaN_xorsign_abs_f16(const unsigned short* a, const unsigned short* b, unsigned short* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  unsigned short r; asm volatile("min.ftz.NaN.xorsign.abs.f16 %0, %1, %2;" : "=h"(r) : "h"(a[i]), "h"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_max_f16(const unsigned short* a, const unsigned short* b, unsigned short* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned short r; asm volatile("max.f16 %0, %1, %2;" : "=h"(r) : "h"(a[i]), "h"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_max_xorsign_abs_f16(const unsigned short* a, const unsigned short* b, unsigned short* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  unsigned short r; asm volatile("max.xorsign.abs.f16 %0, %1, %2;" : "=h"(r) : "h"(a[i]), "h"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_max_NaN_f16(const unsigned short* a, const unsigned short* b, unsigned short* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned short r; asm volatile("max.NaN.f16 %0, %1, %2;" : "=h"(r) : "h"(a[i]), "h"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_max_NaN_xorsign_abs_f16(const unsigned short* a, const unsigned short* b, unsigned short* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  unsigned short r; asm volatile("max.NaN.xorsign.abs.f16 %0, %1, %2;" : "=h"(r) : "h"(a[i]), "h"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_max_ftz_f16(const unsigned short* a, const unsigned short* b, unsigned short* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned short r; asm volatile("max.ftz.f16 %0, %1, %2;" : "=h"(r) : "h"(a[i]), "h"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_max_ftz_xorsign_abs_f16(const unsigned short* a, const unsigned short* b, unsigned short* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  unsigned short r; asm volatile("max.ftz.xorsign.abs.f16 %0, %1, %2;" : "=h"(r) : "h"(a[i]), "h"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_max_ftz_NaN_f16(const unsigned short* a, const unsigned short* b, unsigned short* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned short r; asm volatile("max.ftz.NaN.f16 %0, %1, %2;" : "=h"(r) : "h"(a[i]), "h"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_max_ftz_NaN_xorsign_abs_f16(const unsigned short* a, const unsigned short* b, unsigned short* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  unsigned short r; asm volatile("max.ftz.NaN.xorsign.abs.f16 %0, %1, %2;" : "=h"(r) : "h"(a[i]), "h"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_min_f16x2(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned r; asm volatile("min.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a[i]), "r"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_min_xorsign_abs_f16x2(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  unsigned r; asm volatile("min.xorsign.abs.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a[i]), "r"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_min_NaN_f16x2(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned r; asm volatile("min.NaN.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a[i]), "r"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_min_NaN_xorsign_abs_f16x2(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  unsigned r; asm volatile("min.NaN.xorsign.abs.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a[i]), "r"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_min_ftz_f16x2(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned r; asm volatile("min.ftz.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a[i]), "r"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_min_ftz_xorsign_abs_f16x2(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  unsigned r; asm volatile("min.ftz.xorsign.abs.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a[i]), "r"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_min_ftz_NaN_f16x2(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned r; asm volatile("min.ftz.NaN.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a[i]), "r"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_min_ftz_NaN_xorsign_abs_f16x2(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  unsigned r; asm volatile("min.ftz.NaN.xorsign.abs.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a[i]), "r"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_max_f16x2(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned r; asm volatile("max.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a[i]), "r"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_max_xorsign_abs_f16x2(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  unsigned r; asm volatile("max.xorsign.abs.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a[i]), "r"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_max_NaN_f16x2(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned r; asm volatile("max.NaN.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a[i]), "r"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_max_NaN_xorsign_abs_f16x2(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  unsigned r; asm volatile("max.NaN.xorsign.abs.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a[i]), "r"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_max_ftz_f16x2(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned r; asm volatile("max.ftz.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a[i]), "r"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_max_ftz_xorsign_abs_f16x2(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  unsigned r; asm volatile("max.ftz.xorsign.abs.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a[i]), "r"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_max_ftz_NaN_f16x2(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned r; asm volatile("max.ftz.NaN.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a[i]), "r"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_max_ftz_NaN_xorsign_abs_f16x2(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  unsigned r; asm volatile("max.ftz.NaN.xorsign.abs.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a[i]), "r"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_min_bf16(const unsigned short* a, const unsigned short* b, unsigned short* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned short r; asm volatile("min.bf16 %0, %1, %2;" : "=h"(r) : "h"(a[i]), "h"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_min_xorsign_abs_bf16(const unsigned short* a, const unsigned short* b, unsigned short* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  unsigned short r; asm volatile("min.xorsign.abs.bf16 %0, %1, %2;" : "=h"(r) : "h"(a[i]), "h"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_min_NaN_bf16(const unsigned short* a, const unsigned short* b, unsigned short* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned short r; asm volatile("min.NaN.bf16 %0, %1, %2;" : "=h"(r) : "h"(a[i]), "h"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_min_NaN_xorsign_abs_bf16(const unsigned short* a, const unsigned short* b, unsigned short* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  unsigned short r; asm volatile("min.NaN.xorsign.abs.bf16 %0, %1, %2;" : "=h"(r) : "h"(a[i]), "h"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_max_bf16(const unsigned short* a, const unsigned short* b, unsigned short* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned short r; asm volatile("max.bf16 %0, %1, %2;" : "=h"(r) : "h"(a[i]), "h"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_max_xorsign_abs_bf16(const unsigned short* a, const unsigned short* b, unsigned short* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  unsigned short r; asm volatile("max.xorsign.abs.bf16 %0, %1, %2;" : "=h"(r) : "h"(a[i]), "h"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_max_NaN_bf16(const unsigned short* a, const unsigned short* b, unsigned short* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned short r; asm volatile("max.NaN.bf16 %0, %1, %2;" : "=h"(r) : "h"(a[i]), "h"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_max_NaN_xorsign_abs_bf16(const unsigned short* a, const unsigned short* b, unsigned short* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  unsigned short r; asm volatile("max.NaN.xorsign.abs.bf16 %0, %1, %2;" : "=h"(r) : "h"(a[i]), "h"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_min_bf16x2(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned r; asm volatile("min.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a[i]), "r"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_min_xorsign_abs_bf16x2(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  unsigned r; asm volatile("min.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a[i]), "r"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_min_NaN_bf16x2(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned r; asm volatile("min.NaN.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a[i]), "r"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_min_NaN_xorsign_abs_bf16x2(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  unsigned r; asm volatile("min.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a[i]), "r"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_max_bf16x2(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned r; asm volatile("max.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a[i]), "r"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_max_xorsign_abs_bf16x2(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  unsigned r; asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a[i]), "r"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_max_NaN_bf16x2(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned r; asm volatile("max.NaN.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a[i]), "r"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
__global__ void k_max_NaN_xorsign_abs_bf16x2(const unsigned* a, const unsigned* b, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  unsigned r; asm volatile("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a[i]), "r"(b[i]));
  o[i] = r;
#else
  o[i] = 0;
#endif
}
#if __CUDACC_VER_MAJOR__ >= 13   // PTX ISA 8.8's three-input min/max
__global__ void k_min_f32_3(const unsigned* a, const unsigned* b, const unsigned* c, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  float r; asm volatile("min.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])), "f"(__uint_as_float(c[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
#endif
#if __CUDACC_VER_MAJOR__ >= 13   // PTX ISA 8.8's three-input min/max
__global__ void k_min_abs_f32_3(const unsigned* a, const unsigned* b, const unsigned* c, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  float r; asm volatile("min.abs.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])), "f"(__uint_as_float(c[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
#endif
#if __CUDACC_VER_MAJOR__ >= 13   // PTX ISA 8.8's three-input min/max
__global__ void k_min_NaN_f32_3(const unsigned* a, const unsigned* b, const unsigned* c, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  float r; asm volatile("min.NaN.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])), "f"(__uint_as_float(c[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
#endif
#if __CUDACC_VER_MAJOR__ >= 13   // PTX ISA 8.8's three-input min/max
__global__ void k_min_NaN_abs_f32_3(const unsigned* a, const unsigned* b, const unsigned* c, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  float r; asm volatile("min.NaN.abs.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])), "f"(__uint_as_float(c[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
#endif
#if __CUDACC_VER_MAJOR__ >= 13   // PTX ISA 8.8's three-input min/max
__global__ void k_min_ftz_f32_3(const unsigned* a, const unsigned* b, const unsigned* c, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  float r; asm volatile("min.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])), "f"(__uint_as_float(c[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
#endif
#if __CUDACC_VER_MAJOR__ >= 13   // PTX ISA 8.8's three-input min/max
__global__ void k_min_ftz_abs_f32_3(const unsigned* a, const unsigned* b, const unsigned* c, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  float r; asm volatile("min.ftz.abs.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])), "f"(__uint_as_float(c[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
#endif
#if __CUDACC_VER_MAJOR__ >= 13   // PTX ISA 8.8's three-input min/max
__global__ void k_min_ftz_NaN_f32_3(const unsigned* a, const unsigned* b, const unsigned* c, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  float r; asm volatile("min.ftz.NaN.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])), "f"(__uint_as_float(c[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
#endif
#if __CUDACC_VER_MAJOR__ >= 13   // PTX ISA 8.8's three-input min/max
__global__ void k_min_ftz_NaN_abs_f32_3(const unsigned* a, const unsigned* b, const unsigned* c, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  float r; asm volatile("min.ftz.NaN.abs.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])), "f"(__uint_as_float(c[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
#endif
#if __CUDACC_VER_MAJOR__ >= 13   // PTX ISA 8.8's three-input min/max
__global__ void k_max_f32_3(const unsigned* a, const unsigned* b, const unsigned* c, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  float r; asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])), "f"(__uint_as_float(c[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
#endif
#if __CUDACC_VER_MAJOR__ >= 13   // PTX ISA 8.8's three-input min/max
__global__ void k_max_abs_f32_3(const unsigned* a, const unsigned* b, const unsigned* c, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  float r; asm volatile("max.abs.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])), "f"(__uint_as_float(c[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
#endif
#if __CUDACC_VER_MAJOR__ >= 13   // PTX ISA 8.8's three-input min/max
__global__ void k_max_NaN_f32_3(const unsigned* a, const unsigned* b, const unsigned* c, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  float r; asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])), "f"(__uint_as_float(c[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
#endif
#if __CUDACC_VER_MAJOR__ >= 13   // PTX ISA 8.8's three-input min/max
__global__ void k_max_NaN_abs_f32_3(const unsigned* a, const unsigned* b, const unsigned* c, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  float r; asm volatile("max.NaN.abs.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])), "f"(__uint_as_float(c[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
#endif
#if __CUDACC_VER_MAJOR__ >= 13   // PTX ISA 8.8's three-input min/max
__global__ void k_max_ftz_f32_3(const unsigned* a, const unsigned* b, const unsigned* c, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  float r; asm volatile("max.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])), "f"(__uint_as_float(c[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
#endif
#if __CUDACC_VER_MAJOR__ >= 13   // PTX ISA 8.8's three-input min/max
__global__ void k_max_ftz_abs_f32_3(const unsigned* a, const unsigned* b, const unsigned* c, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  float r; asm volatile("max.ftz.abs.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])), "f"(__uint_as_float(c[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
#endif
#if __CUDACC_VER_MAJOR__ >= 13   // PTX ISA 8.8's three-input min/max
__global__ void k_max_ftz_NaN_f32_3(const unsigned* a, const unsigned* b, const unsigned* c, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  float r; asm volatile("max.ftz.NaN.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])), "f"(__uint_as_float(c[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
#endif
#if __CUDACC_VER_MAJOR__ >= 13   // PTX ISA 8.8's three-input min/max
__global__ void k_max_ftz_NaN_abs_f32_3(const unsigned* a, const unsigned* b, const unsigned* c, unsigned* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  float r; asm volatile("max.ftz.NaN.abs.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(__uint_as_float(a[i])), "f"(__uint_as_float(b[i])), "f"(__uint_as_float(c[i])));
  o[i] = __float_as_uint(r);
#else
  o[i] = 0;
#endif
}
#endif

template <class T> static T* upload(const std::vector<T>& v) {
  T* d; cudaMalloc(&d, v.size() * sizeof(T)); cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice); return d;
}
struct Form { const char* name; const char* type; bool is_min; Mods mods; int nin; int arch; };
static int device_arch() { cudaDeviceProp p; cudaGetDeviceProperties(&p, 0); return p.major * 100 + p.minor * 10; }

// The 16-bit and 32-bit forms: operands in `a`, `b` (and `c`), results to compare.
template <class K2>
static void run2(const Form& f, K2 kernel, Fmt fmt, bool packed) {
  if (device_arch() < f.arch) { ++g_skip; return; }
  ++g_run;
  const auto v = vals(fmt);
  const size_t K = v.size(), N = K * K;
  using T = typename std::conditional<(sizeof(uint32_t) == 4), uint32_t, uint32_t>::type;
  (void)sizeof(T);
  std::vector<uint32_t> A(N), B(N), A2(N), B2(N);
  for (size_t i = 0; i < K; ++i)
    for (size_t j = 0; j < K; ++j) {
      A[i * K + j] = v[i]; B[i * K + j] = v[j];
      A2[i * K + j] = v[(i * 5 + 1) % K]; B2[i * K + j] = v[(j * 3 + 2) % K];   // the second lane of a packed pair
    }
  std::vector<uint32_t> pa(N), pb(N);
  for (size_t i = 0; i < N; ++i) { pa[i] = packed ? (A[i] | (A2[i] << 16)) : A[i]; pb[i] = packed ? (B[i] | (B2[i] << 16)) : B[i]; }
  std::vector<unsigned short> sa(N), sb(N), so(N);
  for (size_t i = 0; i < N; ++i) { sa[i] = (unsigned short)pa[i]; sb[i] = (unsigned short)pb[i]; }
  std::vector<uint32_t> out(N);
  if (fmt.bits == 16 && !packed) {
    unsigned short *da = upload(sa), *db = upload(sb), *dout; cudaMalloc(&dout, N * 2);
    ((void (*)(const unsigned short*, const unsigned short*, unsigned short*, int))kernel)<<<(N + 127) / 128, 128>>>(da, db, dout, (int)N);
    cudaMemcpy(so.data(), dout, N * 2, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < N; ++i) out[i] = so[i];
    cudaFree(da); cudaFree(db); cudaFree(dout);
  } else {
    unsigned *da = upload(pa), *db = upload(pb), *dout; cudaMalloc(&dout, N * 4);
    ((void (*)(const unsigned*, const unsigned*, unsigned*, int))kernel)<<<(N + 127) / 128, 128>>>(da, db, dout, (int)N);
    cudaMemcpy(out.data(), dout, N * 4, cudaMemcpyDeviceToHost);
    cudaFree(da); cudaFree(db); cudaFree(dout);
  }
  cudaDeviceSynchronize();
  int bad = 0;
  for (size_t i = 0; i < N; ++i) {
    for (int lane = 0; lane < (packed ? 2 : 1); ++lane) {
      const uint32_t x = lane ? A2[i] : A[i], y = lane ? B2[i] : B[i];
      const uint32_t got = packed ? (out[i] >> (16 * lane)) & 0xffffu : out[i];
      const Res want = model_min_max(f.is_min, x, y, fmt, f.mods);
      const bool ok = want.any_nan ? is_nan(got, fmt) : got == want.bits;
      if (!ok && bad++ < 3)
        std::printf("FAIL %s: min/max of %0*x and %0*x%s gave %0*x, expected %s%0*x\n", f.name, fmt.bits / 4, x, fmt.bits / 4, y, packed ? (lane ? " (high lane)" : " (low lane)") : "",
                    fmt.bits / 4, got, want.any_nan ? "a NaN, e.g. " : "", fmt.bits / 4, want.any_nan ? (unsigned)0x7fffffffu >> (32 - fmt.bits) : want.bits);
    }
  }
  if (bad) { std::printf("FAIL %s: %d wrong results\n", f.name, bad); ++g_fail; }
}
static void run3(const Form& f, void (*kernel)(const unsigned*, const unsigned*, const unsigned*, unsigned*, int)) {
  if (device_arch() < f.arch) { ++g_skip; return; }
  ++g_run;
  const auto v = vals(F32);
  std::vector<uint32_t> pick;   // 12 operands: every kind once, both signs for a few
  for (size_t i = 0; i < v.size(); i += 2) pick.push_back(v[i]);
  pick.push_back(v[1]); pick.push_back(v[3]); pick.push_back(v[9]);
  const size_t K = pick.size(), N = K * K * K;
  std::vector<uint32_t> A(N), B(N), C(N), out(N);
  for (size_t i = 0; i < N; ++i) { A[i] = pick[i % K]; B[i] = pick[(i / K) % K]; C[i] = pick[i / K / K]; }
  unsigned *da = upload(A), *db = upload(B), *dc = upload(C), *dout; cudaMalloc(&dout, N * 4);
  kernel<<<(N + 127) / 128, 128>>>(da, db, dc, dout, (int)N);
  cudaMemcpy(out.data(), dout, N * 4, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  cudaFree(da); cudaFree(db); cudaFree(dc); cudaFree(dout);
  int bad = 0;
  for (size_t i = 0; i < N; ++i) {
    const Res want = model_min_max3(f.is_min, A[i], B[i], C[i], f.mods);
    const bool ok = want.any_nan ? is_nan(out[i], F32) : out[i] == want.bits;
    if (!ok && bad++ < 3) std::printf("FAIL %s: of %08x %08x %08x gave %08x, expected %s%08x\n", f.name, A[i], B[i], C[i], out[i], want.any_nan ? "a NaN, e.g. " : "", want.any_nan ? 0x7fffffffu : want.bits);
  }
  if (bad) { std::printf("FAIL %s: %d wrong results\n", f.name, bad); ++g_fail; }
}

int main() {
  cudaFree(0);
  run2(Form{"min.f32", "f32", true, Mods{false, false, false, false}, 2, 750}, (void*)k_min_f32, F32, false);
  run2(Form{"min.xorsign.abs.f32", "f32", true, Mods{false, false, true, false}, 2, 860}, (void*)k_min_xorsign_abs_f32, F32, false);
  run2(Form{"min.NaN.f32", "f32", true, Mods{false, true, false, false}, 2, 750}, (void*)k_min_NaN_f32, F32, false);
  run2(Form{"min.NaN.xorsign.abs.f32", "f32", true, Mods{false, true, true, false}, 2, 860}, (void*)k_min_NaN_xorsign_abs_f32, F32, false);
  run2(Form{"min.ftz.f32", "f32", true, Mods{true, false, false, false}, 2, 750}, (void*)k_min_ftz_f32, F32, false);
  run2(Form{"min.ftz.xorsign.abs.f32", "f32", true, Mods{true, false, true, false}, 2, 860}, (void*)k_min_ftz_xorsign_abs_f32, F32, false);
  run2(Form{"min.ftz.NaN.f32", "f32", true, Mods{true, true, false, false}, 2, 750}, (void*)k_min_ftz_NaN_f32, F32, false);
  run2(Form{"min.ftz.NaN.xorsign.abs.f32", "f32", true, Mods{true, true, true, false}, 2, 860}, (void*)k_min_ftz_NaN_xorsign_abs_f32, F32, false);
  run2(Form{"max.f32", "f32", false, Mods{false, false, false, false}, 2, 750}, (void*)k_max_f32, F32, false);
  run2(Form{"max.xorsign.abs.f32", "f32", false, Mods{false, false, true, false}, 2, 860}, (void*)k_max_xorsign_abs_f32, F32, false);
  run2(Form{"max.NaN.f32", "f32", false, Mods{false, true, false, false}, 2, 750}, (void*)k_max_NaN_f32, F32, false);
  run2(Form{"max.NaN.xorsign.abs.f32", "f32", false, Mods{false, true, true, false}, 2, 860}, (void*)k_max_NaN_xorsign_abs_f32, F32, false);
  run2(Form{"max.ftz.f32", "f32", false, Mods{true, false, false, false}, 2, 750}, (void*)k_max_ftz_f32, F32, false);
  run2(Form{"max.ftz.xorsign.abs.f32", "f32", false, Mods{true, false, true, false}, 2, 860}, (void*)k_max_ftz_xorsign_abs_f32, F32, false);
  run2(Form{"max.ftz.NaN.f32", "f32", false, Mods{true, true, false, false}, 2, 750}, (void*)k_max_ftz_NaN_f32, F32, false);
  run2(Form{"max.ftz.NaN.xorsign.abs.f32", "f32", false, Mods{true, true, true, false}, 2, 860}, (void*)k_max_ftz_NaN_xorsign_abs_f32, F32, false);
  run2(Form{"min.f16", "f16", true, Mods{false, false, false, false}, 2, 800}, (void*)k_min_f16, F16, false);
  run2(Form{"min.xorsign.abs.f16", "f16", true, Mods{false, false, true, false}, 2, 860}, (void*)k_min_xorsign_abs_f16, F16, false);
  run2(Form{"min.NaN.f16", "f16", true, Mods{false, true, false, false}, 2, 800}, (void*)k_min_NaN_f16, F16, false);
  run2(Form{"min.NaN.xorsign.abs.f16", "f16", true, Mods{false, true, true, false}, 2, 860}, (void*)k_min_NaN_xorsign_abs_f16, F16, false);
  run2(Form{"min.ftz.f16", "f16", true, Mods{true, false, false, false}, 2, 800}, (void*)k_min_ftz_f16, F16, false);
  run2(Form{"min.ftz.xorsign.abs.f16", "f16", true, Mods{true, false, true, false}, 2, 860}, (void*)k_min_ftz_xorsign_abs_f16, F16, false);
  run2(Form{"min.ftz.NaN.f16", "f16", true, Mods{true, true, false, false}, 2, 800}, (void*)k_min_ftz_NaN_f16, F16, false);
  run2(Form{"min.ftz.NaN.xorsign.abs.f16", "f16", true, Mods{true, true, true, false}, 2, 860}, (void*)k_min_ftz_NaN_xorsign_abs_f16, F16, false);
  run2(Form{"max.f16", "f16", false, Mods{false, false, false, false}, 2, 800}, (void*)k_max_f16, F16, false);
  run2(Form{"max.xorsign.abs.f16", "f16", false, Mods{false, false, true, false}, 2, 860}, (void*)k_max_xorsign_abs_f16, F16, false);
  run2(Form{"max.NaN.f16", "f16", false, Mods{false, true, false, false}, 2, 800}, (void*)k_max_NaN_f16, F16, false);
  run2(Form{"max.NaN.xorsign.abs.f16", "f16", false, Mods{false, true, true, false}, 2, 860}, (void*)k_max_NaN_xorsign_abs_f16, F16, false);
  run2(Form{"max.ftz.f16", "f16", false, Mods{true, false, false, false}, 2, 800}, (void*)k_max_ftz_f16, F16, false);
  run2(Form{"max.ftz.xorsign.abs.f16", "f16", false, Mods{true, false, true, false}, 2, 860}, (void*)k_max_ftz_xorsign_abs_f16, F16, false);
  run2(Form{"max.ftz.NaN.f16", "f16", false, Mods{true, true, false, false}, 2, 800}, (void*)k_max_ftz_NaN_f16, F16, false);
  run2(Form{"max.ftz.NaN.xorsign.abs.f16", "f16", false, Mods{true, true, true, false}, 2, 860}, (void*)k_max_ftz_NaN_xorsign_abs_f16, F16, false);
  run2(Form{"min.f16x2", "f16x2", true, Mods{false, false, false, false}, 2, 800}, (void*)k_min_f16x2, F16, true);
  run2(Form{"min.xorsign.abs.f16x2", "f16x2", true, Mods{false, false, true, false}, 2, 860}, (void*)k_min_xorsign_abs_f16x2, F16, true);
  run2(Form{"min.NaN.f16x2", "f16x2", true, Mods{false, true, false, false}, 2, 800}, (void*)k_min_NaN_f16x2, F16, true);
  run2(Form{"min.NaN.xorsign.abs.f16x2", "f16x2", true, Mods{false, true, true, false}, 2, 860}, (void*)k_min_NaN_xorsign_abs_f16x2, F16, true);
  run2(Form{"min.ftz.f16x2", "f16x2", true, Mods{true, false, false, false}, 2, 800}, (void*)k_min_ftz_f16x2, F16, true);
  run2(Form{"min.ftz.xorsign.abs.f16x2", "f16x2", true, Mods{true, false, true, false}, 2, 860}, (void*)k_min_ftz_xorsign_abs_f16x2, F16, true);
  run2(Form{"min.ftz.NaN.f16x2", "f16x2", true, Mods{true, true, false, false}, 2, 800}, (void*)k_min_ftz_NaN_f16x2, F16, true);
  run2(Form{"min.ftz.NaN.xorsign.abs.f16x2", "f16x2", true, Mods{true, true, true, false}, 2, 860}, (void*)k_min_ftz_NaN_xorsign_abs_f16x2, F16, true);
  run2(Form{"max.f16x2", "f16x2", false, Mods{false, false, false, false}, 2, 800}, (void*)k_max_f16x2, F16, true);
  run2(Form{"max.xorsign.abs.f16x2", "f16x2", false, Mods{false, false, true, false}, 2, 860}, (void*)k_max_xorsign_abs_f16x2, F16, true);
  run2(Form{"max.NaN.f16x2", "f16x2", false, Mods{false, true, false, false}, 2, 800}, (void*)k_max_NaN_f16x2, F16, true);
  run2(Form{"max.NaN.xorsign.abs.f16x2", "f16x2", false, Mods{false, true, true, false}, 2, 860}, (void*)k_max_NaN_xorsign_abs_f16x2, F16, true);
  run2(Form{"max.ftz.f16x2", "f16x2", false, Mods{true, false, false, false}, 2, 800}, (void*)k_max_ftz_f16x2, F16, true);
  run2(Form{"max.ftz.xorsign.abs.f16x2", "f16x2", false, Mods{true, false, true, false}, 2, 860}, (void*)k_max_ftz_xorsign_abs_f16x2, F16, true);
  run2(Form{"max.ftz.NaN.f16x2", "f16x2", false, Mods{true, true, false, false}, 2, 800}, (void*)k_max_ftz_NaN_f16x2, F16, true);
  run2(Form{"max.ftz.NaN.xorsign.abs.f16x2", "f16x2", false, Mods{true, true, true, false}, 2, 860}, (void*)k_max_ftz_NaN_xorsign_abs_f16x2, F16, true);
  run2(Form{"min.bf16", "bf16", true, Mods{false, false, false, false}, 2, 800}, (void*)k_min_bf16, BF16, false);
  run2(Form{"min.xorsign.abs.bf16", "bf16", true, Mods{false, false, true, false}, 2, 860}, (void*)k_min_xorsign_abs_bf16, BF16, false);
  run2(Form{"min.NaN.bf16", "bf16", true, Mods{false, true, false, false}, 2, 800}, (void*)k_min_NaN_bf16, BF16, false);
  run2(Form{"min.NaN.xorsign.abs.bf16", "bf16", true, Mods{false, true, true, false}, 2, 860}, (void*)k_min_NaN_xorsign_abs_bf16, BF16, false);
  run2(Form{"max.bf16", "bf16", false, Mods{false, false, false, false}, 2, 800}, (void*)k_max_bf16, BF16, false);
  run2(Form{"max.xorsign.abs.bf16", "bf16", false, Mods{false, false, true, false}, 2, 860}, (void*)k_max_xorsign_abs_bf16, BF16, false);
  run2(Form{"max.NaN.bf16", "bf16", false, Mods{false, true, false, false}, 2, 800}, (void*)k_max_NaN_bf16, BF16, false);
  run2(Form{"max.NaN.xorsign.abs.bf16", "bf16", false, Mods{false, true, true, false}, 2, 860}, (void*)k_max_NaN_xorsign_abs_bf16, BF16, false);
  run2(Form{"min.bf16x2", "bf16x2", true, Mods{false, false, false, false}, 2, 800}, (void*)k_min_bf16x2, BF16, true);
  run2(Form{"min.xorsign.abs.bf16x2", "bf16x2", true, Mods{false, false, true, false}, 2, 860}, (void*)k_min_xorsign_abs_bf16x2, BF16, true);
  run2(Form{"min.NaN.bf16x2", "bf16x2", true, Mods{false, true, false, false}, 2, 800}, (void*)k_min_NaN_bf16x2, BF16, true);
  run2(Form{"min.NaN.xorsign.abs.bf16x2", "bf16x2", true, Mods{false, true, true, false}, 2, 860}, (void*)k_min_NaN_xorsign_abs_bf16x2, BF16, true);
  run2(Form{"max.bf16x2", "bf16x2", false, Mods{false, false, false, false}, 2, 800}, (void*)k_max_bf16x2, BF16, true);
  run2(Form{"max.xorsign.abs.bf16x2", "bf16x2", false, Mods{false, false, true, false}, 2, 860}, (void*)k_max_xorsign_abs_bf16x2, BF16, true);
  run2(Form{"max.NaN.bf16x2", "bf16x2", false, Mods{false, true, false, false}, 2, 800}, (void*)k_max_NaN_bf16x2, BF16, true);
  run2(Form{"max.NaN.xorsign.abs.bf16x2", "bf16x2", false, Mods{false, true, true, false}, 2, 860}, (void*)k_max_NaN_xorsign_abs_bf16x2, BF16, true);
#if __CUDACC_VER_MAJOR__ >= 13
  run3(Form{"min.f32.3", "f32", true, Mods{false, false, false, false}, 3, 1000}, k_min_f32_3);
#endif
#if __CUDACC_VER_MAJOR__ >= 13
  run3(Form{"min.abs.f32.3", "f32", true, Mods{false, false, false, true}, 3, 1000}, k_min_abs_f32_3);
#endif
#if __CUDACC_VER_MAJOR__ >= 13
  run3(Form{"min.NaN.f32.3", "f32", true, Mods{false, true, false, false}, 3, 1000}, k_min_NaN_f32_3);
#endif
#if __CUDACC_VER_MAJOR__ >= 13
  run3(Form{"min.NaN.abs.f32.3", "f32", true, Mods{false, true, false, true}, 3, 1000}, k_min_NaN_abs_f32_3);
#endif
#if __CUDACC_VER_MAJOR__ >= 13
  run3(Form{"min.ftz.f32.3", "f32", true, Mods{true, false, false, false}, 3, 1000}, k_min_ftz_f32_3);
#endif
#if __CUDACC_VER_MAJOR__ >= 13
  run3(Form{"min.ftz.abs.f32.3", "f32", true, Mods{true, false, false, true}, 3, 1000}, k_min_ftz_abs_f32_3);
#endif
#if __CUDACC_VER_MAJOR__ >= 13
  run3(Form{"min.ftz.NaN.f32.3", "f32", true, Mods{true, true, false, false}, 3, 1000}, k_min_ftz_NaN_f32_3);
#endif
#if __CUDACC_VER_MAJOR__ >= 13
  run3(Form{"min.ftz.NaN.abs.f32.3", "f32", true, Mods{true, true, false, true}, 3, 1000}, k_min_ftz_NaN_abs_f32_3);
#endif
#if __CUDACC_VER_MAJOR__ >= 13
  run3(Form{"max.f32.3", "f32", false, Mods{false, false, false, false}, 3, 1000}, k_max_f32_3);
#endif
#if __CUDACC_VER_MAJOR__ >= 13
  run3(Form{"max.abs.f32.3", "f32", false, Mods{false, false, false, true}, 3, 1000}, k_max_abs_f32_3);
#endif
#if __CUDACC_VER_MAJOR__ >= 13
  run3(Form{"max.NaN.f32.3", "f32", false, Mods{false, true, false, false}, 3, 1000}, k_max_NaN_f32_3);
#endif
#if __CUDACC_VER_MAJOR__ >= 13
  run3(Form{"max.NaN.abs.f32.3", "f32", false, Mods{false, true, false, true}, 3, 1000}, k_max_NaN_abs_f32_3);
#endif
#if __CUDACC_VER_MAJOR__ >= 13
  run3(Form{"max.ftz.f32.3", "f32", false, Mods{true, false, false, false}, 3, 1000}, k_max_ftz_f32_3);
#endif
#if __CUDACC_VER_MAJOR__ >= 13
  run3(Form{"max.ftz.abs.f32.3", "f32", false, Mods{true, false, false, true}, 3, 1000}, k_max_ftz_abs_f32_3);
#endif
#if __CUDACC_VER_MAJOR__ >= 13
  run3(Form{"max.ftz.NaN.f32.3", "f32", false, Mods{true, true, false, false}, 3, 1000}, k_max_ftz_NaN_f32_3);
#endif
#if __CUDACC_VER_MAJOR__ >= 13
  run3(Form{"max.ftz.NaN.abs.f32.3", "f32", false, Mods{true, true, false, true}, 3, 1000}, k_max_ftz_NaN_abs_f32_3);
#endif
  std::printf("%d forms run, %d skipped on this architecture, %d failed\n", g_run, g_skip, g_fail);
  std::printf(g_fail ? "FAIL\n" : "PASS\n");
  return g_fail != 0;
}
