// Tensor-core input rounding in cuBLAS and cuBLASLt (pantheonsim issue #549).
//
// With cublasSetMathMode(CUBLAS_TF32_TENSOR_OP_MATH), or a FAST_TF32 / FAST_16F / FAST_16BF compute type, a
// tensor-core kernel converts every FP32 element of A and B to a narrower format before it multiplies; the
// products are exact and the sums FP32. This program multiplies a matrix of one value x by a matrix of ones (and
// the other way round) and compares the result with k * round(x), where x sits exactly on a rounding tie of the
// format, so the answer shows the format AND its rounding mode:
//   TF32  10 mantissa bits, nearest with ties AWAY from zero   (1 + 2^-11 -> 1 + 2^-10)
//   FP16  10 mantissa bits, nearest even                       (1 + 2^-11 -> 1)
//   BF16  7 mantissa bits, nearest even                        (1 + 2^-8  -> 1)
// Every expectation here was measured on an RTX 3060 (cuBLAS 13.0) and this program passes there; see
// nvidia/src/gemm_input_rounding.hpp for the table. On NVIDIA_TF32_OVERRIDE=0 (the second registration of this
// test) TF32 is off everywhere.
//
// The card runs a tensor-core kernel only from a size on: 16^3 and gemv-like shapes keep the plain FP32 answer.
// (Shapes whose kernel choice differs between routines on the card, such as m=3 with a batch, are left out.)
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

static int g_fail = 0;
static int g_checks = 0;
#define CHECK(cond, ...)                                  \
  do {                                                    \
    ++g_checks;                                           \
    if (!(cond)) {                                        \
      ++g_fail;                                           \
      std::printf("FAIL %s:%d: ", __FILE__, __LINE__);    \
      std::printf(__VA_ARGS__);                           \
      std::printf("\n");                                  \
    }                                                     \
  } while (0)

static float fb(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static uint32_t ub(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

enum Kind { NONE, TF32, FP16, BF16 };
static const char* kname(Kind k) { return k == NONE ? "fp32" : k == TF32 ? "tf32" : k == FP16 ? "fp16" : "bf16"; }

// The formats, written out here independently of the library: x is positive and in range.
static float model(float x, Kind k) {
  switch (k) {
    case TF32: { uint32_t u = ub(x); u += 0x1000u; return fb(u & ~0x1fffu); }                 // ties away
    case FP16: return __half2float(__float2half(x));                                           // ties to even
    case BF16: return __bfloat162float(__float2bfloat16(x));                                   // ties to even
    default: return x;
  }
}

// The probe values: a tie of TF32 and FP16 (ties away and ties to even differ), a value just above it, a tie of
// BF16 and one just above that.
static const float kProbes[] = {1.0f + 0x1p-11f, 1.0f + 0x1p-11f + 0x1p-13f, 1.0f + 0x1p-8f, 1.0f + 0x1p-8f + 0x1p-10f,
                                3.0f + 0x1p-10f};

struct Shape { int m, n, k; bool rounds; const char* name; };
// Whether the card takes a tensor-core kernel (measured; see gemm_input_rounding.hpp).
static const Shape kShapes[] = {{128, 128, 128, true, "128^3"},   {16, 16, 16, false, "16^3"},
                                {1024, 1, 1024, false, "gemv-like n=1"}, {4, 1024, 512, true, "m=4"}};

struct Ctx {
  cublasHandle_t h;
  cublasLtHandle_t lt;
};
// One routine: C = A (m x k) * B (k x n), alpha 1, beta 0, leading dimensions m, k and m.
using Routine = std::function<cublasStatus_t(Ctx&, int m, int n, int k, const float* A, const float* B, float* C)>;

static bool is_tf32_override_off() {
  const char* e = std::getenv("NVIDIA_TF32_OVERRIDE");
  return e && e[0] == '0' && !e[1];
}

// What a routine does to FP32 inputs, given the handle's math mode (the card's rules).
static Kind kind_for_compute(cublasComputeType_t ct, int mode, bool uses_handle_mode) {
  Kind k = NONE;
  if (ct == CUBLAS_COMPUTE_32F_FAST_TF32) k = TF32;
  else if (ct == CUBLAS_COMPUTE_32F_FAST_16F) k = FP16;
  else if (ct == CUBLAS_COMPUTE_32F_FAST_16BF) k = BF16;
  else if (ct != CUBLAS_COMPUTE_32F_PEDANTIC && uses_handle_mode) k = (mode & 3) == 3 ? TF32 : (mode & 3) == 1 ? FP16 : NONE;
  if (k == TF32 && is_tf32_override_off()) k = NONE;
  return k;
}

static void run_one(Ctx& c, const char* label, const Routine& r, Kind expect) {
  for (const Shape& s : kShapes) {
    for (int side = 0; side < 2; ++side) {
      for (float x : kProbes) {
        const size_t na = (size_t)s.m * s.k, nb = (size_t)s.k * s.n, nc = (size_t)s.m * s.n;
        std::vector<float> a(na, side == 0 ? x : 1.0f), b(nb, side == 0 ? 1.0f : x), out(nc, -1.0f);
        float *dA, *dB, *dC;
        cudaMalloc(&dA, na * 4);
        cudaMalloc(&dB, nb * 4);
        cudaMalloc(&dC, nc * 4);
        cudaMemcpy(dA, a.data(), na * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, b.data(), nb * 4, cudaMemcpyHostToDevice);
        const cublasStatus_t st = r(c, s.m, s.n, s.k, dA, dB, dC);
        cudaDeviceSynchronize();
        cudaMemcpy(out.data(), dC, nc * 4, cudaMemcpyDeviceToHost);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        const Kind k = s.rounds ? expect : NONE;
        const float want = (float)s.k * model(x, k);
        size_t bad = 0;
        for (float v : out) bad += ub(v) != ub(want);
        CHECK(st == CUBLAS_STATUS_SUCCESS && bad == 0,
              "%s, %s, %s side, x=%.9g: expected %s rounding (every element %.9g), got c[0]=%.9g (%zu of %zu elements differ), status %d",
              label, s.name, side == 0 ? "A" : "B", (double)x, kname(k), (double)want, (double)out[0], bad, nc, (int)st);
      }
    }
  }
}

static const cublasComputeType_t kComputes[] = {CUBLAS_COMPUTE_32F, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_COMPUTE_32F_FAST_TF32,
                                                CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_COMPUTE_32F_FAST_16BF};
static const char* cname(cublasComputeType_t ct) {
  switch (ct) {
    case CUBLAS_COMPUTE_32F: return "32F";
    case CUBLAS_COMPUTE_32F_PEDANTIC: return "32F_PEDANTIC";
    case CUBLAS_COMPUTE_32F_FAST_TF32: return "32F_FAST_TF32";
    case CUBLAS_COMPUTE_32F_FAST_16F: return "32F_FAST_16F";
    default: return "32F_FAST_16BF";
  }
}

int main() {
  Ctx c;
  if (cublasCreate(&c.h) || cublasLtCreate(&c.lt)) { std::printf("FAIL: cannot create handles\nFAIL\n"); return 1; }
  const float one = 1.0f, zero = 0.0f;

  const Routine sgemm = [&](Ctx& c, int m, int n, int k, const float* A, const float* B, float* C) {
    return cublasSgemm(c.h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, A, m, B, k, &zero, C, m);
  };
  const Routine sgemm_ex = [&](Ctx& c, int m, int n, int k, const float* A, const float* B, float* C) {
    return cublasSgemmEx(c.h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, A, CUDA_R_32F, m, B, CUDA_R_32F, k, &zero, C, CUDA_R_32F, m);
  };
  const Routine strided = [&](Ctx& c, int m, int n, int k, const float* A, const float* B, float* C) {
    return cublasSgemmStridedBatched(c.h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, A, m, 0, B, k, 0, &zero, C, m, 0, 2);
  };
  const Routine batched = [&](Ctx& c, int m, int n, int k, const float* A, const float* B, float* C) {
    const float* Ap[2] = {A, A};
    const float* Bp[2] = {B, B};
    float* Cp[2] = {C, C};
    const float **dAp, **dBp;
    float** dCp;
    cudaMalloc(&dAp, sizeof Ap); cudaMalloc(&dBp, sizeof Bp); cudaMalloc(&dCp, sizeof Cp);
    cudaMemcpy(dAp, Ap, sizeof Ap, cudaMemcpyHostToDevice);
    cudaMemcpy(dBp, Bp, sizeof Bp, cudaMemcpyHostToDevice);
    cudaMemcpy(dCp, Cp, sizeof Cp, cudaMemcpyHostToDevice);
    const cublasStatus_t st = cublasSgemmBatched(c.h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, dAp, m, dBp, k, &zero, dCp, m, 2);
    cudaDeviceSynchronize();
    cudaFree(dAp); cudaFree(dBp); cudaFree(dCp);
    return st;
  };

  struct Mode { const char* name; cublasMath_t mode; };
  const Mode modes[] = {{"DEFAULT", CUBLAS_DEFAULT_MATH}, {"TF32_TENSOR_OP", CUBLAS_TF32_TENSOR_OP_MATH},
                        {"PEDANTIC", CUBLAS_PEDANTIC_MATH}, {"TENSOR_OP (deprecated)", CUBLAS_TENSOR_OP_MATH}};
  for (const Mode& md : modes) {
    cublasSetMathMode(c.h, md.mode);
    cublasMath_t got;
    cublasGetMathMode(c.h, &got);
    CHECK(got == md.mode, "math mode reads back %d, set %d", (int)got, (int)md.mode);
    const Kind plain = kind_for_compute(CUBLAS_COMPUTE_32F, (int)md.mode, true);
    char label[160];
    std::snprintf(label, sizeof label, "mode %s, Sgemm", md.name);
    run_one(c, label, sgemm, plain);
    std::snprintf(label, sizeof label, "mode %s, SgemmEx", md.name);
    run_one(c, label, sgemm_ex, plain);
    std::snprintf(label, sizeof label, "mode %s, SgemmStridedBatched", md.name);
    run_one(c, label, strided, plain);
    std::snprintf(label, sizeof label, "mode %s, SgemmBatched", md.name);
    run_one(c, label, batched, plain);
    for (cublasComputeType_t ct : kComputes) {
      const Kind want = kind_for_compute(ct, (int)md.mode, true);
      std::snprintf(label, sizeof label, "mode %s, GemmEx %s", md.name, cname(ct));
      run_one(c, label, [&, ct](Ctx& c, int m, int n, int k, const float* A, const float* B, float* C) {
        return cublasGemmEx(c.h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, A, CUDA_R_32F, m, B, CUDA_R_32F, k, &zero, C, CUDA_R_32F, m,
                            ct, CUBLAS_GEMM_DEFAULT);
      }, want);
      std::snprintf(label, sizeof label, "mode %s, GemmStridedBatchedEx %s", md.name, cname(ct));
      run_one(c, label, [&, ct](Ctx& c, int m, int n, int k, const float* A, const float* B, float* C) {
        return cublasGemmStridedBatchedEx(c.h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, A, CUDA_R_32F, m, 0, B, CUDA_R_32F, k, 0, &zero,
                                          C, CUDA_R_32F, m, 0, 2, ct, CUBLAS_GEMM_DEFAULT);
      }, want);
    }
  }

  // cublasLt: the handle's math mode is not read, only the descriptor's compute type.
  cublasSetMathMode(c.h, CUBLAS_TF32_TENSOR_OP_MATH);
  for (cublasComputeType_t ct : kComputes) {
    const Kind want = kind_for_compute(ct, 0, false);
    char label[160];
    std::snprintf(label, sizeof label, "cublasLtMatmul %s", cname(ct));
    run_one(c, label, [&, ct](Ctx& c, int m, int n, int k, const float* A, const float* B, float* C) {
      cublasLtMatmulDesc_t d;
      cublasLtMatrixLayout_t la, lb, lc;
      cublasLtMatmulDescCreate(&d, ct, CUDA_R_32F);
      cublasLtMatrixLayoutCreate(&la, CUDA_R_32F, m, k, m);
      cublasLtMatrixLayoutCreate(&lb, CUDA_R_32F, k, n, k);
      cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, m, n, m);
      void* ws;
      const size_t wsz = 32u << 20;
      cudaMalloc(&ws, wsz);
      cublasLtMatmulPreference_t pr;
      cublasLtMatmulPreferenceCreate(&pr);
      cublasLtMatmulPreferenceSetAttribute(pr, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof wsz);
      cublasLtMatmulHeuristicResult_t hr;
      int nres = 0;
      cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(c.lt, d, la, lb, lc, lc, pr, 1, &hr, &nres);
      if (st == CUBLAS_STATUS_SUCCESS && nres > 0)
        st = cublasLtMatmul(c.lt, d, &one, A, la, B, lb, &zero, C, lc, C, lc, &hr.algo, ws, wsz, 0);
      else if (st == CUBLAS_STATUS_SUCCESS) st = CUBLAS_STATUS_NOT_SUPPORTED;
      cudaDeviceSynchronize();
      cudaFree(ws);
      cublasLtMatmulPreferenceDestroy(pr);
      cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc);
      cublasLtMatmulDescDestroy(d);
      return st;
    }, want);
  }

  // Complex single precision: both parts of every element, Cgemm in TF32 mode and GemmEx(C_32F) with FAST_TF32.
  // (256^3: this is the size the Cgemm measurement was made at.)
  {
    const int n = 256;
    for (int mode = 0; mode < 2; ++mode) {
      cublasSetMathMode(c.h, mode ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
      for (int viaEx = 0; viaEx < 2; ++viaEx) {
        const float x = kProbes[0];
        std::vector<cuComplex> a((size_t)n * n, make_cuComplex(x, x)), b((size_t)n * n, make_cuComplex(1.0f, 0.0f)), out((size_t)n * n);
        cuComplex *dA, *dB, *dC;
        cudaMalloc(&dA, a.size() * 8); cudaMalloc(&dB, b.size() * 8); cudaMalloc(&dC, out.size() * 8);
        cudaMemcpy(dA, a.data(), a.size() * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, b.data(), b.size() * 8, cudaMemcpyHostToDevice);
        const cuComplex o = make_cuComplex(1, 0), z = make_cuComplex(0, 0);
        cublasStatus_t st;
        Kind want;
        if (viaEx) {
          st = cublasGemmEx(c.h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &o, dA, CUDA_C_32F, n, dB, CUDA_C_32F, n, &z, dC, CUDA_C_32F, n,
                            CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
          want = is_tf32_override_off() ? NONE : TF32;
        } else {
          st = cublasCgemm(c.h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &o, dA, n, dB, n, &z, dC, n);
          want = (mode && !is_tf32_override_off()) ? TF32 : NONE;
        }
        cudaDeviceSynchronize();
        cudaMemcpy(out.data(), dC, out.size() * 8, cudaMemcpyDeviceToHost);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        const float w = (float)n * model(x, want);
        size_t bad = 0;
        for (const cuComplex& v : out) bad += ub(v.x) != ub(w) || ub(v.y) != ub(w);
        CHECK(st == CUBLAS_STATUS_SUCCESS && bad == 0, "%s, %s mode: expected %s rounding (%.9g), got %.9g+%.9gi (%zu differ), status %d",
              viaEx ? "GemmEx C_32F FAST_TF32" : "Cgemm", mode ? "TF32" : "default", kname(want), (double)w, (double)out[0].x,
              (double)out[0].y, bad, (int)st);
      }
    }
  }

  // C is never rounded: beta * C keeps its full precision under TF32 (alpha = 0 so the product drops out).
  {
    cublasSetMathMode(c.h, CUBLAS_TF32_TENSOR_OP_MATH);
    const int n = 128;
    const float x = kProbes[0], beta = 1.0f;
    std::vector<float> a((size_t)n * n, 1.0f), b((size_t)n * n, 1.0f), cc((size_t)n * n, x), out(cc.size());
    float *dA, *dB, *dC;
    cudaMalloc(&dA, a.size() * 4); cudaMalloc(&dB, b.size() * 4); cudaMalloc(&dC, cc.size() * 4);
    cudaMemcpy(dA, a.data(), a.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, b.data(), b.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, cc.data(), cc.size() * 4, cudaMemcpyHostToDevice);
    const float zero_alpha = 0.0f;
    cublasSgemm(c.h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &zero_alpha, dA, n, dB, n, &beta, dC, n);
    cudaDeviceSynchronize();
    cudaMemcpy(out.data(), dC, out.size() * 4, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    size_t bad = 0;
    for (float v : out) bad += ub(v) != ub(x);
    CHECK(bad == 0, "beta*C under TF32: C was changed (%zu elements)", bad);
  }

  cublasLtDestroy(c.lt);
  cublasDestroy(c.h);
  std::printf("%d checks, %d failed\n", g_checks, g_fail);
  std::printf(g_fail ? "FAIL\n" : "PASS\n");
  return g_fail != 0;
}
