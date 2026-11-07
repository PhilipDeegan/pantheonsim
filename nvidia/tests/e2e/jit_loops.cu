// Loops and branches for tier 2 (the JIT, src/sass/jit.hpp): kernels whose hot code is a loop, so that the generated code has to run
// uniform branches, leave at a divergent one, count its own steps, honour the 256-instruction run, and go back to the tier at the exact op.
// run_sass_jit.sh runs it with the tier and with the JIT forced on at once; every result and every counter must be the same.
//
//   count    a uniform counted loop (loads, float arithmetic) -- the branch is taken by the whole warp
//   collatz  every lane leaves at its own iteration -- a branch that splits the group ends the generated code, the tier runs it
//   nested   two loops and an early exit (a lane that leaves leaves the inner loop for the outer)
//   spin     a loop on a uniform float test (the lanes agree), with a store in the body (the store gate)
//   unroll   a long unrolled body (more than a region's cap, and more than the 256-op run once the loop's iterations are counted)
//   clocks   the clock around a loop: S2R of the clock reads the warp's step count, so the generated code must have written every counter by then
//   shared   a loop with shared memory and barriers (BAR ends a region)
//   guarded  a loop whose body has guarded ops (@P) and a predicated exit, in a block of 33 threads (a warp of one lane)
//   params   uniform values (ULDC, UIADD3, UISETP...) that steer a loop
//   f64      64-bit accumulators (CS2R of SRZ zeroes a register pair) in a loop
//   race     two warps increment one word without atomics in a loop of more calls than a turn has (4096): where the turns end, and so the number of calls the
//            generated code charges to the turn, decides which updates are lost; the first store of each warp ends its turn (the store gate)
//   race2    the same with a body of about 300 dependent ops between the load and the store: the 256-op run the loop gives a call decides where the warps swap
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at %d\n", cudaGetErrorString(e_), __LINE__); std::exit(1); } } while (0)

__global__ void k_count(float* out, const float* in, int n, int trips) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  float acc = 0.f;
#pragma unroll 1
  for (int i = 0; i < trips; ++i) acc = acc * 0.99f + in[(t + i * 7) % n] * static_cast<float>(i & 15);
  out[t] = acc;
}

__global__ void k_collatz(int* out, int n) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t x = static_cast<uint32_t>(t) + 1u;
  int c = 0;
  while (x != 1u && c < 400) {
    x = (x & 1u) ? 3u * x + 1u : x >> 1;
    ++c;
  }
  if (t < n) out[t] = c;
}

__global__ void k_nested(int* out, int a, int b) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  int s = t;
  for (int i = 0; i < a; ++i) {
    for (int j = 0; j < b; ++j) {
      s += (i * j) ^ t;
      if (s > 5000 + (t & 7)) goto done;
    }
    s ^= i;
  }
done:
  out[t] = s;
}

__global__ void k_spin(float* out, float limit, int stride) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  float v = 1.0f;
  int k = 0;
  while (v < limit) {   // the same for every lane
    v = v * 1.37f + 0.5f;
    out[t + stride * (k & 3)] = v;   // a store in the body (the thread's own cells)
    ++k;
  }
  out[t] = v + static_cast<float>(k);
}

__global__ void k_unroll(float* out, const float* in, int rounds) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  float a = in[t & 63], b = in[(t + 1) & 63], c = in[(t + 2) & 63], d = 0.f;
#pragma unroll 1
  for (int r = 0; r < rounds; ++r) {
#pragma unroll
    for (int u = 0; u < 90; ++u) {
      a = a * 1.0001f + b;
      b = b * 0.9999f - c * 0.5f;
      c = c * 1.0002f + static_cast<float>(u);
      d += a * 0.001f;
    }
  }
  out[t] = a + b + c + d;
}

__global__ void k_clocks(long long* out, int trips) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const long long t0 = clock64();
  int s = t;
#pragma unroll 1
  for (int i = 0; i < trips; ++i) s = s * 5 + i;
  const long long t1 = clock64();
  out[t] = (t1 - t0) * 1000003LL + s;
}

__global__ void k_shared(int* out, int rounds) {
  __shared__ int buf[128];
  const int t = threadIdx.x;
  buf[t] = t * 3 + blockIdx.x;
  __syncthreads();
  int s = 0;
#pragma unroll 1
  for (int r = 0; r < rounds; ++r) {
    s += buf[(t + r) & 127] * (r + 1);
    __syncthreads();
    buf[t] = s;
    __syncthreads();
  }
  out[blockIdx.x * blockDim.x + t] = s;
}

__global__ void k_guarded(int* out, int n, int mod) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  int s = 0, i = 0;
  for (;;) {
    if ((t + i) % mod == 0) s += i * 3;
    else s ^= t;
    if (s < 0 && i > n) break;
    ++i;
    if (i >= n + (t & 3)) break;
  }
  out[t] = s;
}

__global__ void k_params(int* out, int a, int b, int c) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  int s = 0;
  const int step = (a > b) ? (a - b) : (b - a) + 1;
  for (int i = a; i != c; i += step) {
    s += i * (b ^ 5);
    if (s > 1000000) break;
  }
  out[t] = s + t;
}

__global__ void k_f64(double* out, int trips) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  double acc = 0.0;
  long long z = 0;
#pragma unroll 1
  for (int i = 0; i < trips; ++i) {
    acc = acc * 0.5 + static_cast<double>(i + t);
    z += i * 3LL;
  }
  out[t] = acc + static_cast<double>(z);
}

__global__ void k_race(volatile int* p, int iters) {
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    const int x = p[blockIdx.x];
    p[blockIdx.x] = x + 1;
  }
}

__global__ void k_race2(volatile unsigned* p, int iters) {
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    unsigned x = p[blockIdx.x];
    unsigned y = x ^ static_cast<unsigned>(i);
#pragma unroll
    for (int u = 0; u < 60; ++u) {   // (xorshift steps: no closed form for the compiler to fold, about 5 ops each)
      y ^= y << 13;
      y ^= y >> 17;
      y ^= y << 5;
      y += static_cast<unsigned>(u);
    }
    p[blockIdx.x] = x + 1u + (y == 0x12345678u);
  }
}

template <class T>
static uint64_t hash_dev(const T* d, size_t n) {
  std::vector<T> h(n);
  CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
  uint64_t x = 1469598103934665603ull;
  for (size_t i = 0; i < n * sizeof(T); ++i) x = (x ^ reinterpret_cast<const uint8_t*>(h.data())[i]) * 1099511628211ull;
  return x;
}

int main() {
  const int N = 4096;
  float *f_in, *f_out;
  int* i_out;
  long long* l_out;
  double* d_out;
  CK(cudaMalloc(&d_out, N * sizeof(double)));
  CK(cudaMalloc(&f_in, N * sizeof(float)));
  CK(cudaMalloc(&f_out, N * 4 * sizeof(float)));
  CK(cudaMalloc(&i_out, N * sizeof(int)));
  CK(cudaMalloc(&l_out, N * sizeof(long long)));
  std::vector<float> h(N);
  for (int i = 0; i < N; ++i) h[i] = static_cast<float>((i * 37) % 101) * 0.25f - 7.f;
  CK(cudaMemcpy(f_in, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  const int blocks[] = {32, 64, 96, 33, 1};
  for (int bs : blocks) {
    const int grid = bs == 1 ? 3 : 24;
    CK(cudaMemset(f_out, 0, N * 4 * sizeof(float)));
    k_count<<<grid, bs>>>(f_out, f_in, N, 40);
    CK(cudaDeviceSynchronize());
    std::printf("count bs=%d %016llx\n", bs, static_cast<unsigned long long>(hash_dev(f_out, grid * bs)));
    CK(cudaMemset(i_out, 0, N * sizeof(int)));
    k_collatz<<<grid, bs>>>(i_out, grid * bs);
    CK(cudaDeviceSynchronize());
    std::printf("collatz bs=%d %016llx\n", bs, static_cast<unsigned long long>(hash_dev(i_out, grid * bs)));
    k_nested<<<grid, bs>>>(i_out, 12, 9);
    CK(cudaDeviceSynchronize());
    std::printf("nested bs=%d %016llx\n", bs, static_cast<unsigned long long>(hash_dev(i_out, grid * bs)));
    k_spin<<<grid, bs>>>(f_out, 3000.f, grid * bs);
    CK(cudaDeviceSynchronize());
    std::printf("spin bs=%d %016llx\n", bs, static_cast<unsigned long long>(hash_dev(f_out, grid * bs * 4)));
    k_unroll<<<grid, bs>>>(f_out, f_in, 6);
    CK(cudaDeviceSynchronize());
    std::printf("unroll bs=%d %016llx\n", bs, static_cast<unsigned long long>(hash_dev(f_out, grid * bs)));
    k_clocks<<<grid, bs>>>(l_out, 25);
    CK(cudaDeviceSynchronize());
    std::printf("clocks bs=%d %016llx\n", bs, static_cast<unsigned long long>(hash_dev(l_out, grid * bs)));
    k_guarded<<<grid, bs>>>(i_out, 30, 5);
    CK(cudaDeviceSynchronize());
    std::printf("guarded bs=%d %016llx\n", bs, static_cast<unsigned long long>(hash_dev(i_out, grid * bs)));
    k_f64<<<grid, bs>>>(d_out, 17);
    CK(cudaDeviceSynchronize());
    std::printf("f64 bs=%d %016llx\n", bs, static_cast<unsigned long long>(hash_dev(d_out, grid * bs)));
    k_params<<<grid, bs>>>(i_out, 3, 11, 500);
    CK(cudaDeviceSynchronize());
    std::printf("params bs=%d %016llx\n", bs, static_cast<unsigned long long>(hash_dev(i_out, grid * bs)));
  }
  CK(cudaMemset(i_out, 0, 64 * sizeof(int)));
  k_race<<<3, 64>>>(reinterpret_cast<volatile int*>(i_out), 6000);
  CK(cudaDeviceSynchronize());
  std::printf("race %016llx\n", static_cast<unsigned long long>(hash_dev(i_out, 64)));
  CK(cudaMemset(i_out, 0, 64 * sizeof(int)));
  k_race2<<<2, 64>>>(reinterpret_cast<volatile unsigned*>(i_out), 3000);
  CK(cudaDeviceSynchronize());
  std::printf("race2 %016llx\n", static_cast<unsigned long long>(hash_dev(i_out, 64)));
  k_shared<<<8, 128>>>(i_out, 7);
  CK(cudaDeviceSynchronize());
  std::printf("shared %016llx\n", static_cast<unsigned long long>(hash_dev(i_out, 8 * 128)));
  std::printf("PASS\n");
  return 0;
}
