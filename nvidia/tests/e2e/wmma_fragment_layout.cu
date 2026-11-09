// Which matrix element each fragment element is, for every WMMA shape and type, measured the way a program can: load a matrix whose
// element (row, col) holds its own index through nvcuda::wmma::load_matrix_sync (in both memory layouts) and read every element of the
// fragment back (`frag.x[i]`). The PTX leaves the distribution unspecified, but real code relies on what the hardware does: tiny-cuda-nn
// masks an accumulator with the elements of an A fragment (`acc.x[i] *= (a.x[i] > 0)`), CUTLASS epilogues and many hand-written kernels
// do the same. The expected distributions below were measured on an RTX 3060 (sm_86) and are written out here as lists, a different
// way than the simulator's table (include/vgpu/ptx/ast.hpp, wmma_hw_slot), so a mistake in one does not hide in the other. The same program
// runs on a card, where it passes.
//
// Then the pattern that broke tiny-cuda-nn on the simulator: an accumulator (f16 and f32) produced by mma_sync, multiplied element by element
// with a value taken from an A fragment loaded from a matrix of 0s and 1s (the ReLU mask), and stored. Prints PASS on the last line.
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>
#include <utility>

using namespace nvcuda;

static int g_fail = 0;
using RC = std::pair<int, int>;
using List = std::vector<RC>;

// ---- what the fragment elements of lane l should be, as lists of (row, col) ------------------------------------------------
static List cycle_to(const List& base, size_t n) {
  List out;
  for (size_t i = 0; i < n; ++i) out.push_back(base[i % base.size()]);
  return out;
}
// kind: 'a', 'b', 'c'. type: 'h' (f16/bf16), 'i' (s8/u8), 't' (tf32), 'n' (s4/u4), 'o' (b1), 'd' (f64).
static List expected(char kind, char type, int M, int N, int lane, size_t n) {
  const int g = lane / 4, q = lane % 4;
  List b;
  if (kind == 'a') {
    if (type == 'h') {
      if (M == 8) { for (int kh = 0; kh < 2; ++kh) for (int e = 0; e < 2; ++e) b.push_back({g, 8 * kh + 2 * q + e}); }
      else for (int rb = 0; rb < M / 16; ++rb) for (int kh = 0; kh < 2; ++kh) for (int h = 0; h < 2; ++h) for (int e = 0; e < 2; ++e)
        b.push_back({16 * rb + g + 8 * h, 8 * kh + 2 * q + e});
    } else if (type == 'i') {
      if (M == 8) { for (int e = 0; e < 4; ++e) b.push_back({g, 4 * q + e}); }
      else for (int rb = 0; rb < M / 16; ++rb) for (int h = 0; h < 2; ++h) for (int e = 0; e < 4; ++e) b.push_back({16 * rb + g + 8 * h, 4 * q + e});
    } else if (type == 't') {
      for (int kh = 0; kh < 2; ++kh) for (int h = 0; h < 2; ++h) b.push_back({g + 8 * h, 4 * kh + q});
    } else if (type == 'n') { for (int e = 0; e < 8; ++e) b.push_back({g, 8 * q + e}); }
    else if (type == 'o') { for (int e = 0; e < 32; ++e) b.push_back({g, 32 * q + e}); }
    else b.push_back({g, q});
  } else if (kind == 'b') {
    if (type == 'h') {
      if (N == 8) { for (int kh = 0; kh < 2; ++kh) for (int e = 0; e < 2; ++e) b.push_back({8 * kh + 2 * q + e, g}); }
      else if (N == 16) { for (int nh = 0; nh < 2; ++nh) for (int kh = 0; kh < 2; ++kh) for (int e = 0; e < 2; ++e) b.push_back({8 * kh + 2 * q + e, 8 * nh + g}); }
      else for (int nb = 0; nb < 2; ++nb) for (int kh = 0; kh < 2; ++kh) for (int nh = 0; nh < 2; ++nh) for (int e = 0; e < 2; ++e)
        b.push_back({8 * kh + 2 * q + e, 16 * nb + 8 * nh + g});
    } else if (type == 'i') {
      for (int nh = 0; nh < N / 8; ++nh) for (int e = 0; e < 4; ++e) b.push_back({4 * q + e, 8 * nh + g});
    } else if (type == 't') {
      for (int nh = 0; nh < 2; ++nh) for (int kh = 0; kh < 2; ++kh) b.push_back({4 * kh + q, 8 * nh + g});
    } else if (type == 'n') { for (int e = 0; e < 8; ++e) b.push_back({8 * q + e, g}); }
    else if (type == 'o') { for (int e = 0; e < 32; ++e) b.push_back({32 * q + e, g}); }
    else b.push_back({q, g});
  } else {
    if (M == 8 && N == 8) { for (int e = 0; e < 2; ++e) b.push_back({g, 2 * q + e}); }
    else if (M == 16) { for (int nh = 0; nh < 2; ++nh) for (int h = 0; h < 2; ++h) for (int e = 0; e < 2; ++e) b.push_back({g + 8 * h, 8 * nh + 2 * q + e}); }
    else if (M == 32) { for (int rh = 0; rh < 4; ++rh) for (int e = 0; e < 2; ++e) b.push_back({8 * rh + g, 2 * q + e}); }
    else { for (int ch = 0; ch < 4; ++ch) for (int e = 0; e < 2; ++e) b.push_back({2 * q + e, 8 * ch + g}); }
  }
  return cycle_to(b, n);
}

// ---- probing --------------------------------------------------------------------------------------------------------------
__device__ uint64_t bits_of(__half v) { return __half_as_ushort(v); }
__device__ uint64_t bits_of(__nv_bfloat16 v) { return __bfloat16_as_ushort(v); }
__device__ uint64_t bits_of(float v) { return __float_as_uint(v); }
__device__ uint64_t bits_of(double v) { return (uint64_t)__double_as_longlong(v); }
__device__ uint64_t bits_of(signed char v) { return (unsigned char)v; }
__device__ uint64_t bits_of(unsigned char v) { return v; }
__device__ uint64_t bits_of(int v) { return (unsigned)v; }
__device__ uint64_t bits_of(unsigned v) { return v; }

constexpr int kMaxEl = 40;
template <class Use, int M, int N, int K, class T, class L, class Mem>
__global__ void probe_ab(const Mem* mem, int ld, uint64_t* out) {
  wmma::fragment<Use, M, N, K, T, L> f;
  wmma::load_matrix_sync(f, mem, ld);
  for (int t = 0; t < (int)f.num_elements && t < kMaxEl; ++t) out[threadIdx.x * kMaxEl + t] = bits_of(f.x[t]);
  if (threadIdx.x == 0) out[32 * kMaxEl] = f.num_elements;
}
template <int M, int N, int K, class T>
__global__ void probe_c(const T* mem, int ld, wmma::layout_t lay, uint64_t* out) {
  wmma::fragment<wmma::accumulator, M, N, K, T> f;
  wmma::load_matrix_sync(f, mem, ld, lay);
  for (int t = 0; t < (int)f.num_elements && t < kMaxEl; ++t) out[threadIdx.x * kMaxEl + t] = bits_of(f.x[t]);
  if (threadIdx.x == 0) out[32 * kMaxEl] = f.num_elements;
}

// Fills the memory image of a rows x cols matrix with element (r, c) = (index >> shift) & mask, `bits` per element.
static void fill(std::vector<uint8_t>& host, int rows, int cols, bool colmem, int bits, int shift, int pass_bits) {
  const uint64_t pm = pass_bits >= 64 ? ~0ull : (1ull << pass_bits) - 1;
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) {
      const uint64_t idx = colmem ? (uint64_t)c * rows + r : (uint64_t)r * cols + c;
      const uint64_t v = (idx >> shift) & pm;
      if (bits == 64) ((uint64_t*)host.data())[idx] = v;
      else if (bits == 32) ((uint32_t*)host.data())[idx] = (uint32_t)v;
      else if (bits == 16) ((uint16_t*)host.data())[idx] = (uint16_t)v;
      else if (bits == 8) host[idx] = (uint8_t)v;
      else if (bits == 4) host[idx / 2] |= (uint8_t)(v << ((idx & 1) * 4));
      else host[idx / 8] |= (uint8_t)(v << (idx & 7));
    }
}

// Reads the passes back into element indices, per lane: a list of (row, col).
static List decode(const std::vector<std::vector<uint64_t>>& got, int lane, int n_el, int rows, int cols, bool colmem, int elem_bits, int pass_bits, bool packed) {
  List out;
  // (a sub-byte fragment's num_elements counts elements but only the first word is loaded: the register holds 32 / elem_bits of them)
  if (packed) n_el = 1;
  const int per_word = packed ? 32 / elem_bits : 1;
  for (int t = 0; t < n_el; ++t)
    for (int e = 0; e < per_word; ++e) {
      uint64_t idx = 0;
      for (size_t p = 0; p < got.size(); ++p) {
        uint64_t w = got[p][lane * kMaxEl + t];
        if (packed) w >>= e * elem_bits;
        const uint64_t pm = pass_bits >= 64 ? ~0ull : (1ull << pass_bits) - 1;
        idx |= (w & pm) << (p * pass_bits);
      }
      out.push_back(colmem ? RC{(int)(idx % rows), (int)(idx / rows)} : RC{(int)(idx / cols), (int)(idx % cols)});
    }
  return out;
}

static bool run_probe(const char* name, char kind, char type, int M, int N, int K, bool colmem, int rows, int cols, int elem_bits, int pass_bits,
                      int passes, bool packed, const void* kernel_tag, void (*launch)(const void*, int, uint64_t*)) {
  (void)kernel_tag;
  std::vector<std::vector<uint64_t>> got;
  int n_el = 0;
  for (int p = 0; p < passes; ++p) {
    std::vector<uint8_t> host((size_t)rows * cols * 8 + 8, 0);
    fill(host, rows, cols, colmem, elem_bits, p * pass_bits, pass_bits);
    void* dm; uint64_t* dout;
    cudaMalloc(&dm, host.size()); cudaMalloc(&dout, (32 * kMaxEl + 1) * 8);
    cudaMemset(dout, 0, (32 * kMaxEl + 1) * 8);
    cudaMemcpy(dm, host.data(), host.size(), cudaMemcpyHostToDevice);
    launch(dm, colmem ? rows : cols, dout);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("FAIL %s: %s\n", name, cudaGetErrorString(e)); ++g_fail; return false; }
    std::vector<uint64_t> h(32 * kMaxEl + 1);
    cudaMemcpy(h.data(), dout, h.size() * 8, cudaMemcpyDeviceToHost);
    n_el = (int)h[32 * kMaxEl];
    got.push_back(std::move(h));
    cudaFree(dm); cudaFree(dout);
  }
  int bad = 0;
  for (int lane = 0; lane < 32 && bad < 3; ++lane) {
    const List have = decode(got, lane, n_el, rows, cols, colmem, elem_bits, pass_bits, packed);
    const List want = expected(kind, type, M, N, lane, have.size());
    if (have != want) {
      ++bad;
      printf("FAIL %s (%s memory) lane %d:\n  have", name, colmem ? "column-major" : "row-major", lane);
      for (auto& x : have) printf(" %d,%d", x.first, x.second);
      printf("\n  want");
      for (auto& x : want) printf(" %d,%d", x.first, x.second);
      printf("\n");
    }
  }
  if (bad) ++g_fail;
  return !bad;
}

// One A-or-B configuration in both memory layouts (sub-byte types: the one layout the API allows, A row-major and B column-major).
template <class Use, int M, int N, int K, class T, class Mem, int ELEM_BITS, int PASS_BITS, int PASSES, bool PACKED>
static void ab(const char* name, char kind, char type, int rows, int cols) {
  struct Launch {
    static void row(const void* m, int ld, uint64_t* o) {
      if constexpr (!PACKED || std::is_same_v<Use, wmma::matrix_a>)
        probe_ab<Use, M, N, K, T, wmma::row_major, Mem><<<1, 32>>>((const Mem*)m, ld, o);
    }
    static void col(const void* m, int ld, uint64_t* o) {
      if constexpr (!PACKED || std::is_same_v<Use, wmma::matrix_b>)
        probe_ab<Use, M, N, K, T, wmma::col_major, Mem><<<1, 32>>>((const Mem*)m, ld, o);
    }
  };
  if (!PACKED || std::is_same_v<Use, wmma::matrix_a>)
    run_probe(name, kind, type, M, N, K, false, rows, cols, ELEM_BITS, PASS_BITS, PASSES, PACKED, nullptr, Launch::row);
  if (!PACKED || std::is_same_v<Use, wmma::matrix_b>)
    run_probe(name, kind, type, M, N, K, true, rows, cols, ELEM_BITS, PASS_BITS, PASSES, PACKED, nullptr, Launch::col);
}
template <int M, int N, int K, class T, int BITS>
static void acc(const char* name) {
  for (int colmem = 0; colmem < 2; ++colmem) {
    const int rows = M, cols = N;
    std::vector<uint8_t> host((size_t)rows * cols * 8, 0);
    fill(host, rows, cols, colmem, BITS, 0, BITS);
    void* dm; uint64_t* dout;
    cudaMalloc(&dm, host.size()); cudaMalloc(&dout, (32 * kMaxEl + 1) * 8); cudaMemset(dout, 0, (32 * kMaxEl + 1) * 8);
    cudaMemcpy(dm, host.data(), host.size(), cudaMemcpyHostToDevice);
    probe_c<M, N, K, T><<<1, 32>>>((const T*)dm, colmem ? M : N, colmem ? wmma::mem_col_major : wmma::mem_row_major, dout);
    cudaError_t e = cudaDeviceSynchronize();
    std::vector<uint64_t> h(32 * kMaxEl + 1);
    cudaMemcpy(h.data(), dout, h.size() * 8, cudaMemcpyDeviceToHost);
    cudaFree(dm); cudaFree(dout);
    if (e != cudaSuccess) { printf("FAIL %s: %s\n", name, cudaGetErrorString(e)); ++g_fail; return; }
    std::vector<std::vector<uint64_t>> got{h};
    int bad = 0;
    for (int lane = 0; lane < 32 && bad < 3; ++lane) {
      const List have = decode(got, lane, (int)h[32 * kMaxEl], rows, cols, colmem, BITS, BITS, false);
      const List want = expected('c', 'x', M, N, lane, have.size());
      if (have != want) {
        ++bad;
        printf("FAIL %s accumulator (%s memory) lane %d:\n  have", name, colmem ? "column-major" : "row-major", lane);
        for (auto& x : have) printf(" %d,%d", x.first, x.second);
        printf("\n  want");
        for (auto& x : want) printf(" %d,%d", x.first, x.second);
        printf("\n");
      }
    }
    if (bad) ++g_fail;
  }
}

// ---- the ReLU-mask pattern -------------------------------------------------------------------------------------------------
// D = A x B (acc of type TC), then D.x[i] *= (mask.x[i] > 0) where mask is an A fragment loaded from a matrix of 0s and 1s, stored.
template <class TC>
__global__ void relu_mask_kernel(const __half* a, const __half* b, const __half* mask, TC* out) {
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa, fm;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
  wmma::fragment<wmma::accumulator, 16, 16, 16, TC> acc;
  wmma::load_matrix_sync(fa, a, 16);
  wmma::load_matrix_sync(fb, b, 16);
  wmma::load_matrix_sync(fm, mask, 16);
  wmma::fill_fragment(acc, (TC)0.0f);
  wmma::mma_sync(acc, fa, fb, acc);
  for (int t = 0; t < (int)acc.num_elements; ++t) acc.x[t] = acc.x[t] * (TC)(float)((float)fm.x[t] > 0.0f);
  wmma::store_matrix_sync(out, acc, 16, wmma::mem_row_major);
}
template <class TC>
static void relu_mask(const char* name) {
  std::vector<__half> a(256), b(256), m(256);
  std::vector<float> ref(256, 0.0f);
  // small integers: every product and sum is exact in f16 and f32
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j) {
      a[i * 16 + j] = __float2half((float)((i * 3 + j * 5) % 7 - 3));
      b[i * 16 + j] = __float2half((float)((i * 5 + j * 2) % 5 - 2));
      m[i * 16 + j] = __float2half(((i * 7 + j * 3 + (i >> 2)) % 3 == 0) ? 0.0f : 1.0f);   // a mask that depends on the row and the column
    }
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j) {
      float s = 0;
      for (int k = 0; k < 16; ++k) s += __half2float(a[i * 16 + k]) * __half2float(b[k * 16 + j]);
      ref[i * 16 + j] = __half2float(m[i * 16 + j]) > 0.0f ? s : 0.0f;
    }
  __half *da, *db, *dm; TC* dout;
  cudaMalloc(&da, 512); cudaMalloc(&db, 512); cudaMalloc(&dm, 512); cudaMalloc(&dout, 256 * sizeof(TC));
  cudaMemcpy(da, a.data(), 512, cudaMemcpyHostToDevice); cudaMemcpy(db, b.data(), 512, cudaMemcpyHostToDevice); cudaMemcpy(dm, m.data(), 512, cudaMemcpyHostToDevice);
  relu_mask_kernel<TC><<<1, 32>>>(da, db, dm, dout);
  cudaError_t e = cudaDeviceSynchronize();
  std::vector<TC> out(256);
  cudaMemcpy(out.data(), dout, 256 * sizeof(TC), cudaMemcpyDeviceToHost);
  cudaFree(da); cudaFree(db); cudaFree(dm); cudaFree(dout);
  int bad = 0;
  for (int i = 0; i < 256; ++i) if ((float)out[i] != ref[i]) ++bad;
  if (e != cudaSuccess || bad) { printf("FAIL %s: %d of 256 elements wrong (%s)\n", name, bad, cudaGetErrorString(e)); ++g_fail; }
  else printf("ok   %s\n", name);
}

int main() {
  using namespace wmma;
  // f16 and bf16: three shapes
  ab<matrix_a, 16, 16, 16, __half, __half, 16, 16, 1, false>("A f16 m16n16k16", 'a', 'h', 16, 16);
  ab<matrix_a, 32, 8, 16, __half, __half, 16, 16, 1, false>("A f16 m32n8k16", 'a', 'h', 32, 16);
  ab<matrix_a, 8, 32, 16, __half, __half, 16, 16, 1, false>("A f16 m8n32k16", 'a', 'h', 8, 16);
  ab<matrix_b, 16, 16, 16, __half, __half, 16, 16, 1, false>("B f16 m16n16k16", 'b', 'h', 16, 16);
  ab<matrix_b, 32, 8, 16, __half, __half, 16, 16, 1, false>("B f16 m32n8k16", 'b', 'h', 16, 8);
  ab<matrix_b, 8, 32, 16, __half, __half, 16, 16, 1, false>("B f16 m8n32k16", 'b', 'h', 16, 32);
  ab<matrix_a, 16, 16, 16, __nv_bfloat16, __nv_bfloat16, 16, 16, 1, false>("A bf16 m16n16k16", 'a', 'h', 16, 16);
  ab<matrix_a, 32, 8, 16, __nv_bfloat16, __nv_bfloat16, 16, 16, 1, false>("A bf16 m32n8k16", 'a', 'h', 32, 16);
  ab<matrix_a, 8, 32, 16, __nv_bfloat16, __nv_bfloat16, 16, 16, 1, false>("A bf16 m8n32k16", 'a', 'h', 8, 16);
  ab<matrix_b, 16, 16, 16, __nv_bfloat16, __nv_bfloat16, 16, 16, 1, false>("B bf16 m16n16k16", 'b', 'h', 16, 16);
  ab<matrix_b, 32, 8, 16, __nv_bfloat16, __nv_bfloat16, 16, 16, 1, false>("B bf16 m32n8k16", 'b', 'h', 16, 8);
  ab<matrix_b, 8, 32, 16, __nv_bfloat16, __nv_bfloat16, 16, 16, 1, false>("B bf16 m8n32k16", 'b', 'h', 16, 32);
  ab<matrix_a, 16, 16, 8, precision::tf32, float, 32, 32, 1, false>("A tf32 m16n16k8", 'a', 't', 16, 8);
  ab<matrix_b, 16, 16, 8, precision::tf32, float, 32, 32, 1, false>("B tf32 m16n16k8", 'b', 't', 8, 16);
  ab<matrix_a, 8, 8, 4, double, double, 64, 64, 1, false>("A f64 m8n8k4", 'a', 'd', 8, 4);
  ab<matrix_b, 8, 8, 4, double, double, 64, 64, 1, false>("B f64 m8n8k4", 'b', 'd', 4, 8);
  // 4-bit and single-bit (A row-major, B column-major only); several elements per register word
  ab<matrix_a, 8, 8, 32, experimental::precision::s4, void, 4, 4, 2, true>("A s4 m8n8k32", 'a', 'n', 8, 32);
  ab<matrix_b, 8, 8, 32, experimental::precision::s4, void, 4, 4, 2, true>("B s4 m8n8k32", 'b', 'n', 32, 8);
  ab<matrix_a, 8, 8, 32, experimental::precision::u4, void, 4, 4, 2, true>("A u4 m8n8k32", 'a', 'n', 8, 32);
  ab<matrix_a, 8, 8, 128, experimental::precision::b1, void, 1, 1, 10, true>("A b1 m8n8k128", 'a', 'o', 8, 128);
  ab<matrix_b, 8, 8, 128, experimental::precision::b1, void, 1, 1, 10, true>("B b1 m8n8k128", 'b', 'o', 128, 8);
  // 8-bit integers: three shapes (the index is two bytes, so two passes)
  ab<matrix_a, 16, 16, 16, signed char, signed char, 8, 8, 2, false>("A s8 m16n16k16", 'a', 'i', 16, 16);
  ab<matrix_a, 32, 8, 16, signed char, signed char, 8, 8, 2, false>("A s8 m32n8k16", 'a', 'i', 32, 16);
  ab<matrix_a, 8, 32, 16, signed char, signed char, 8, 8, 2, false>("A s8 m8n32k16", 'a', 'i', 8, 16);
  ab<matrix_b, 16, 16, 16, signed char, signed char, 8, 8, 2, false>("B s8 m16n16k16", 'b', 'i', 16, 16);
  ab<matrix_b, 32, 8, 16, signed char, signed char, 8, 8, 2, false>("B s8 m32n8k16", 'b', 'i', 16, 8);
  ab<matrix_b, 8, 32, 16, signed char, signed char, 8, 8, 2, false>("B s8 m8n32k16", 'b', 'i', 16, 32);
  ab<matrix_a, 16, 16, 16, unsigned char, unsigned char, 8, 8, 2, false>("A u8 m16n16k16", 'a', 'i', 16, 16);
  ab<matrix_b, 16, 16, 16, unsigned char, unsigned char, 8, 8, 2, false>("B u8 m16n16k16", 'b', 'i', 16, 16);
  // accumulators
  acc<16, 16, 16, __half, 16>("C f16 m16n16k16");
  acc<32, 8, 16, __half, 16>("C f16 m32n8k16");
  acc<8, 32, 16, __half, 16>("C f16 m8n32k16");
  acc<16, 16, 16, float, 32>("C f32 m16n16k16");
  acc<32, 8, 16, float, 32>("C f32 m32n8k16");
  acc<8, 32, 16, float, 32>("C f32 m8n32k16");
  acc<16, 16, 16, int, 32>("C s32 m16n16k16");
  acc<8, 8, 4, double, 64>("C f64 m8n8k4");
  relu_mask<__half>("relu mask on an f16 accumulator");
  relu_mask<float>("relu mask on an f32 accumulator");
  if (g_fail) { printf("%d FAILED\n", g_fail); return 1; }
  printf("PASS\n");
  return 0;
}
