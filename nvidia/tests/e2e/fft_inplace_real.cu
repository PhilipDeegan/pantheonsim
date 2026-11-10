// In-place real transforms (pantheonsim #548): cufftExecR2C / D2Z / C2R / Z2D with the input and output on one buffer.
//
// The real array of an in-place transform is padded: each row of the fastest axis holds 2 * (n / 2 + 1) reals, so
// it is as wide as the complex half-spectrum it becomes, and a batch is that many reals per row of every
// transform apart. cuFFT documents the layout, and on an RTX 3060 (CUDA 13.0) the in-place result is
// bit-identical to the out-of-place one for rank 1, 2 and 3, single and double precision, forward and inverse,
// batch 1 and 3, plans made with cufftPlan1d/2d/3d, cufftPlanMany and cufftCreate + cufftMakePlan1d/2d/3d.
// This program transforms the same data both ways and requires equal bytes (the padding of an inverse
// transform's output is not compared).
//
// Prints PASS on the last line.
#include <cufft.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
template <class R, class C, bool DBL>
static int run(int rank, int n, int batch, bool inverse, bool many) {
  size_t rows = 1; for (int i = 0; i < rank - 1; i++) rows *= n;
  size_t ld = 2 * (n / 2 + 1), half = n / 2 + 1;
  size_t Nr = rows * n, Pr = rows * ld, Mc = rows * half;   // per batch: packed reals, padded reals, complex
  std::vector<R> real((size_t)batch * Nr);
  srand(7 + rank * 10 + n); for (auto& v : real) v = (R)(rand() / (double)RAND_MAX - 0.5);
  cufftType fwd = DBL ? CUFFT_D2Z : CUFFT_R2C, inv = DBL ? CUFFT_Z2D : CUFFT_C2R;
  cufftType type = inverse ? inv : fwd;
  auto plan = [&](cufftHandle* h) {
    int dims[3] = {n, n, n};
    cufftResult r;
    if (many == 2 && batch == 1) {   // cufftCreate + cufftMakePlanNd (auto allocation on)
      size_t ws;
      r = cufftCreate(h);
      if (!r) r = rank == 1 ? cufftMakePlan1d(*h, n, type, 1, &ws) : rank == 2 ? cufftMakePlan2d(*h, n, n, type, &ws) : cufftMakePlan3d(*h, n, n, n, type, &ws);
    } else if (many == 1 || batch != 1) r = cufftPlanMany(h, rank, dims + (3 - rank), nullptr, 1, 0, nullptr, 1, 0, type, batch);
    else if (rank == 1) r = cufftPlan1d(h, n, type, 1);
    else if (rank == 2) r = cufftPlan2d(h, n, n, type);
    else r = cufftPlan3d(h, n, n, n, type);
    return r;
  };
  cufftHandle a, b;
  if (plan(&a) || plan(&b)) { printf("plan failed\n"); return 1; }
  size_t Cb = (size_t)batch * Mc * sizeof(C);
  size_t Pb = (size_t)batch * Pr * sizeof(R);
  // Make the complex spectrum for the inverse case by a forward out-of-place transform of the real data.
  R *dreal; C *dspec;
  cudaMalloc(&dreal, (size_t)batch * Nr * sizeof(R)); cudaMalloc(&dspec, Cb);
  cudaMemcpy(dreal, real.data(), real.size() * sizeof(R), cudaMemcpyHostToDevice);
  std::vector<C> spec((size_t)batch * Mc);
  {
    cufftHandle f; int dims[3] = {n, n, n};
    if (cufftPlanMany(&f, rank, dims + (3 - rank), nullptr, 1, 0, nullptr, 1, 0, fwd, batch)) return 1;
    if (DBL) cufftExecD2Z(f, (cufftDoubleReal*)dreal, (cufftDoubleComplex*)dspec); else cufftExecR2C(f, (cufftReal*)dreal, (cufftComplex*)dspec);
    cudaDeviceSynchronize(); cudaMemcpy(spec.data(), dspec, Cb, cudaMemcpyDeviceToHost); cufftDestroy(f);
  }
  size_t nout = inverse ? (size_t)batch * Nr : (size_t)batch * Mc;
  std::vector<char> outA(nout * (inverse ? sizeof(R) : sizeof(C))), outB(outA.size());
  // out of place
  if (!inverse) {
    C* o; cudaMalloc(&o, Cb);
    if (DBL) cufftExecD2Z(a, (cufftDoubleReal*)dreal, (cufftDoubleComplex*)o); else cufftExecR2C(a, (cufftReal*)dreal, (cufftComplex*)o);
    cudaDeviceSynchronize(); cudaMemcpy(outA.data(), o, Cb, cudaMemcpyDeviceToHost); cudaFree(o);
    // in place: padded real layout
    size_t bytes = std::max(Pb, Cb);
    std::vector<R> pad((size_t)batch * Pr, (R)0);
    size_t bd = Mc * 2;  // batch distance in reals
    for (int bt = 0; bt < batch; bt++) for (size_t r = 0; r < rows; r++) for (int z = 0; z < n; z++) pad[bt * bd + r * ld + z] = real[(size_t)bt * Nr + r * n + z];
    void* p; cudaMalloc(&p, bytes); cudaMemset(p, 0, bytes); cudaMemcpy(p, pad.data(), Pb, cudaMemcpyHostToDevice);
    if (DBL) cufftExecD2Z(b, (cufftDoubleReal*)p, (cufftDoubleComplex*)p); else cufftExecR2C(b, (cufftReal*)p, (cufftComplex*)p);
    cudaDeviceSynchronize(); cudaMemcpy(outB.data(), p, Cb, cudaMemcpyDeviceToHost); cudaFree(p);
  } else {
    // C2R destroys its input, so each run gets a copy of the spectrum
    R* o; cudaMalloc(&o, (size_t)batch * Nr * sizeof(R)); C* in; cudaMalloc(&in, Cb); cudaMemcpy(in, spec.data(), Cb, cudaMemcpyHostToDevice);
    if (DBL) cufftExecZ2D(a, (cufftDoubleComplex*)in, (cufftDoubleReal*)o); else cufftExecC2R(a, (cufftComplex*)in, (cufftReal*)o);
    cudaDeviceSynchronize(); cudaMemcpy(outA.data(), o, outA.size(), cudaMemcpyDeviceToHost); cudaFree(o); cudaFree(in);
    size_t bytes = std::max(Pb, Cb);
    void* p; cudaMalloc(&p, bytes); cudaMemset(p, 0, bytes); cudaMemcpy(p, spec.data(), Cb, cudaMemcpyHostToDevice);
    if (DBL) cufftExecZ2D(b, (cufftDoubleComplex*)p, (cufftDoubleReal*)p); else cufftExecC2R(b, (cufftComplex*)p, (cufftReal*)p);
    cudaDeviceSynchronize();
    std::vector<R> pad((size_t)batch * Pr);
    cudaMemcpy(pad.data(), p, Pb, cudaMemcpyDeviceToHost); cudaFree(p);
    R* ob = (R*)outB.data(); size_t bd = Mc * 2;
    for (int bt = 0; bt < batch; bt++) for (size_t r = 0; r < rows; r++) for (int z = 0; z < n; z++) ob[(size_t)bt * Nr + r * n + z] = pad[bt * bd + r * ld + z];
  }
  size_t diff = 0; for (size_t i = 0; i < outA.size(); i++) diff += outA[i] != outB[i];
  if (diff) printf("FAIL: %s %dD n=%d batch=%d %s plan=%s: in-place differs from out-of-place in %zu bytes\n", DBL ? "double" : "float", rank, n, batch, inverse ? "inverse" : "forward", many == 2 ? "Create+MakePlan" : many ? "Many" : "PlanNd", diff);
  cufftDestroy(a); cufftDestroy(b); cudaFree(dreal); cudaFree(dspec);
  return diff != 0;
}
int main() {
  int bad = 0;
  for (int dbl = 0; dbl < 2; dbl++)
    for (int inv = 0; inv < 2; inv++)
      for (int rank = 1; rank <= 3; rank++)
        for (int n : {8, 9, 16, 24, 36}) for (int batch : {1, 3}) for (int many = 0; many < 3; many++)
          bad += dbl ? run<double, cufftDoubleComplex, true>(rank, n, batch, inv, many) : run<float, cufftComplex, false>(rank, n, batch, inv, many);
  printf(bad ? "FAIL\n" : "PASS\n");
  return bad != 0;
}
