// sens <mem|compute> <launches> <elements> [iters]: integer-only kernels (exact set), checksum printed.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>
__global__ void memk(const uint32_t* in, uint32_t* out, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) out[i] = in[i] * 2654435761u + (uint32_t)i;
}
__global__ void compk(uint32_t* out, int n, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t x = i * 2246822519u + 1;
  for (int k = 0; k < iters; ++k) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; x = x * 3266489917u + 374761393u; }
  out[i] = x;
}
int main(int argc, char** argv) {
  bool mem = !strcmp(argv[1], "mem"); int L = atoi(argv[2]); int n = atoi(argv[3]); int iters = argc > 4 ? atoi(argv[4]) : 256;
  uint32_t *a, *b; cudaMalloc(&a, n * 4); cudaMalloc(&b, n * 4);
  uint32_t* h = (uint32_t*)malloc(n * 4); for (int i = 0; i < n; ++i) h[i] = i * 7u;
  cudaMemcpy(a, h, n * 4, cudaMemcpyHostToDevice);
  int blocks = (n + 255) / 256; if (blocks > 65535) blocks = 65535;
  for (int l = 0; l < L; ++l) {
    if (mem) memk<<<blocks, 256>>>(a, b, n); else compk<<<(n + 255) / 256, 256>>>(b, n, iters);
    if (mem) { uint32_t* t = a; a = b; b = t; }
  }
  cudaDeviceSynchronize();
  cudaMemcpy(h, mem ? a : b, n * 4, cudaMemcpyDeviceToHost);
  uint64_t s = 0; for (int i = 0; i < n; ++i) s = s * 1099511628211ull + h[i];
  printf("checksum %016llx\n", (unsigned long long)s);
  return 0;
}
