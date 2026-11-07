// Launch-level result caching (VGPU_MEMO), through the CUDA runtime API: every way a program changes device memory or
// launches work, and a checksum of what the device holds after each step. run_memo_api_matrix.sh runs this program with
// the cache off, cold and warm and requires the same text from all three, hits on the warm run, and no difference when
// every hit is also run for real (VGPU_MEMO_VERIFY). Steps that the cache cannot represent (managed and pinned memory,
// device printf, device malloc, cooperative launches, textures, atomics) are here too: they must run and be correct, and
// must leave later launches uncached rather than wrong.
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

__constant__ unsigned c_scale[4];
__device__ unsigned g_acc[4];

__global__ void axpy(unsigned* y, const unsigned* x, unsigned a, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = y[i] * a + x[i];
}
__global__ void fixed7(unsigned* y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = 7;
}
__global__ void use_const(unsigned* y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = y[i] * c_scale[i & 3] + 1;
}
__global__ void bump_global(unsigned v) { g_acc[threadIdx.x & 3] += v; }
__global__ void read_global(unsigned* y) { y[threadIdx.x] = g_acc[threadIdx.x & 3] + threadIdx.x; }
__global__ void atomic_count(unsigned* y) { atomicAdd(y, 1u); }
__global__ void print_one(const unsigned* y) { if (threadIdx.x == 0 && blockIdx.x == 0) printf("device says %u\n", y[0]); }
__global__ void heap_user(unsigned* y) {
  unsigned* p = (unsigned*)malloc(16);
  if (p) { p[0] = 41; y[threadIdx.x] = p[0] + threadIdx.x; free(p); }
}
__global__ void coop(unsigned* y, int n) {
  namespace cg = cooperative_groups;
  cg::grid_group g = cg::this_grid();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = i;
  g.sync();
  if (i < n) y[i] += y[(i + 1) % n];
}
__global__ void tex_sum(cudaTextureObject_t t, unsigned* y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = tex1Dfetch<unsigned>(t, i) + 1;
}
__global__ void managed_inc(unsigned* m, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) m[i] += 3;
}

static unsigned long long sum_dev(const unsigned* d, int n) {
  std::vector<unsigned> h(n);
  CK(cudaMemcpy(h.data(), d, n * sizeof(unsigned), cudaMemcpyDeviceToHost));
  unsigned long long s = 1469598103934665603ull;
  for (int i = 0; i < n; ++i) s = (s ^ h[i]) * 1099511628211ull;
  return s;
}
#define STEP(name, d, n) printf("%-28s %016llx\n", name, sum_dev(d, n))

int main() {
  const int N = 1024;
  unsigned *y, *x, *z;
  CK(cudaMalloc(&y, N * 4));
  CK(cudaMalloc(&x, N * 4));
  CK(cudaMalloc(&z, N * 4));
  std::vector<unsigned> h(N);
  for (int i = 0; i < N; ++i) h[i] = i * 3 + 1;
  cudaStream_t s1, s2;
  CK(cudaStreamCreate(&s1));
  CK(cudaStreamCreate(&s2));

  // Host to device, every flavour; memset, every width.
  CK(cudaMemcpy(x, h.data(), N * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpyAsync(z, h.data(), N * 4, cudaMemcpyHostToDevice, s1));
  CK(cudaStreamSynchronize(s1));
  CK(cudaMemset(y, 1, N * 4));
  STEP("h2d + memset", y, N);
  CK(cudaMemset2D(z, 256, 0x02, 64, 8));
  STEP("memset2D", z, N);
  CK(cudaMemset(y, 0, N * 4));
  CK(cudaMemcpy2D(z, 256, x, 256, 128, 8, cudaMemcpyDeviceToDevice));
  STEP("memcpy2D d2d", z, N);

  // Launches on a state that changes, then on one that does not.
  for (int i = 0; i < 3; ++i) axpy<<<4, 256>>>(y, x, 2, N);
  STEP("axpy x3", y, N);
  for (int i = 0; i < 4; ++i) fixed7<<<4, 256>>>(y, N);
  STEP("fixed7 x4", y, N);

  // Device to device.
  CK(cudaMemcpy(z, y, N * 4, cudaMemcpyDeviceToDevice));
  CK(cudaMemcpyAsync(x, z, N * 2, cudaMemcpyDeviceToDevice, s2));
  CK(cudaStreamSynchronize(s2));
  axpy<<<4, 256>>>(z, x, 3, N);
  STEP("d2d + axpy", z, N);

  // Symbols: constant and device variables.
  const unsigned scale[4] = {2, 3, 4, 5};
  CK(cudaMemcpyToSymbol(c_scale, scale, sizeof scale));
  use_const<<<4, 256>>>(y, N);
  STEP("constant bank", y, N);
  bump_global<<<1, 32>>>(5);
  bump_global<<<1, 32>>>(5);
  read_global<<<1, 64>>>(z);
  STEP("device variable", z, 64);

  // Two streams, interleaved: the order the program issues them in is the history.
  axpy<<<4, 256, 0, s1>>>(y, x, 2, N);
  axpy<<<4, 256, 0, s2>>>(y, x, 5, N);
  axpy<<<4, 256, 0, s1>>>(y, x, 3, N);
  CK(cudaDeviceSynchronize());
  STEP("two streams", y, N);

  // A graph, captured and replayed.
  cudaGraph_t g;
  cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal));
  CK(cudaMemsetAsync(z, 0, N * 4, s1));
  axpy<<<4, 256, 0, s1>>>(z, x, 1, N);
  fixed7<<<4, 256, 0, s1>>>(y, N);
  CK(cudaStreamEndCapture(s1, &g));
  CK(cudaGraphInstantiate(&ge, g, 0));
  for (int i = 0; i < 3; ++i) CK(cudaGraphLaunch(ge, s1));
  CK(cudaStreamSynchronize(s1));
  STEP("graph x3", z, N);
  STEP("graph x3 (y)", y, N);

  // Allocation and free, stream-ordered too.
  unsigned* t;
  CK(cudaMallocAsync(&t, N * 4, s1));
  CK(cudaMemsetAsync(t, 9, N * 4, s1));
  axpy<<<4, 256, 0, s1>>>(t, x, 2, N);
  CK(cudaStreamSynchronize(s1));
  STEP("mallocAsync", t, N);
  CK(cudaFreeAsync(t, s1));
  CK(cudaStreamSynchronize(s1));
  unsigned* pitched;
  size_t pitch;
  CK(cudaMallocPitch((void**)&pitched, &pitch, 100, 8));
  CK(cudaMemset(pitched, 3, pitch * 8));
  printf("pitch %zu\n", pitch);
  CK(cudaFree(pitched));

  // What the cache cannot represent. Each runs, is checked, and leaves later launches uncached.
  CK(cudaMemset(z, 0, N * 4));
  for (int i = 0; i < 4; ++i) atomic_count<<<4, 64>>>(z);
  STEP("atomics", z, 4);
  print_one<<<1, 1>>>(x);
  CK(cudaDeviceSynchronize());
  heap_user<<<1, 8>>>(z);
  STEP("device malloc", z, 8);
  {
    void* args[] = {&z, (void*)&N};
    int nn = 256;
    args[1] = &nn;
    CK(cudaLaunchCooperativeKernel((void*)coop, dim3(2), dim3(128), args));
    STEP("cooperative", z, 256);
  }
  {
    cudaResourceDesc rd = {};
    rd.resType = cudaResourceTypeLinear;
    rd.res.linear.devPtr = x;
    rd.res.linear.desc = cudaCreateChannelDesc<unsigned>();
    rd.res.linear.sizeInBytes = N * 4;
    cudaTextureDesc td = {};
    td.readMode = cudaReadModeElementType;
    cudaTextureObject_t tex;
    CK(cudaCreateTextureObject(&tex, &rd, &td, nullptr));
    tex_sum<<<4, 256>>>(tex, y, N);
    STEP("texture", y, N);
    CK(cudaDestroyTextureObject(tex));
  }
  {
    unsigned* m;
    CK(cudaMallocManaged(&m, N * 4));
    for (int i = 0; i < N; ++i) m[i] = i;
    managed_inc<<<4, 256>>>(m, N);
    CK(cudaDeviceSynchronize());
    unsigned long long s = 0;
    for (int i = 0; i < N; ++i) s += m[i];
    printf("%-28s %llu\n", "managed", s);
    CK(cudaFree(m));
  }
  {
    unsigned* hp;
    CK(cudaHostAlloc((void**)&hp, N * 4, cudaHostAllocMapped));
    for (int i = 0; i < N; ++i) hp[i] = i * 2;
    unsigned* dp;
    CK(cudaHostGetDevicePointer((void**)&dp, hp, 0));
    managed_inc<<<4, 256>>>(dp, N);
    CK(cudaDeviceSynchronize());
    unsigned long long s = 0;
    for (int i = 0; i < N; ++i) s += hp[i];
    printf("%-28s %llu\n", "pinned mapped", s);
    CK(cudaFreeHost(hp));
  }
  // After all of that, ordinary work again.
  CK(cudaMemset(y, 0, N * 4));
  for (int i = 0; i < 3; ++i) axpy<<<4, 256>>>(y, x, 2, N);
  STEP("after: axpy x3", y, N);
  CK(cudaFree(y));
  CK(cudaFree(x));
  CK(cudaFree(z));
  printf("PASS\n");
  return 0;
}
