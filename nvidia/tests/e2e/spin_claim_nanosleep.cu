// Lanes of one warp race for a slot with atomicCAS; the losers spin, with __nanosleep as the back-off, until the
// winner publishes. ptxas compiles the loop to
//   BSSY B0 ; @P1 BRA past ; NANOSLEEP ; (VOTEU, FLO, one lane's ATOMG.ADD, SHFL) ; @!P1 BRA loop ; BSYNC B0
// with no YIELD: the winner stands at the BSYNC closing the loop and waits for the lanes that spin because it has
// not yet published. Real hardware (independent thread scheduling) runs it, as a NANOSLEEP lets the other threads
// go; the simulator ran the spinners for ever (exec-limit) because only a YIELD made a group give way. This is the
// shape of eliphatfs/cuacd's mesh_audit.cuh edge hash table insert (pantheongpu/pantheonsim#553).
#include <cstdio>
#include <cuda_runtime.h>
__global__ void k(unsigned* slot, unsigned* out) {
  unsigned s = blockIdx.x;
  unsigned old = atomicCAS(&slot[s], 0u, 0xFFFFFFFFu);
  if (old == 0xFFFFFFFFu) {
    do { __nanosleep(16); old = atomicAdd(&slot[s], 0u); } while (old == 0xFFFFFFFFu);
  }
  if (old == 0u) atomicExch(&slot[s], 42u);   // the winner publishes
  out[blockIdx.x * blockDim.x + threadIdx.x] = old + 1;   // the winner saw 0, the others the published 42
}
int main() {
  int bad = 0;
  for (int threads : {32, 33, 64, 128}) {
    const int blocks = 3;
    unsigned *slot, *out;
    cudaMalloc(&slot, blocks * 4); cudaMemset(slot, 0, blocks * 4);
    cudaMalloc(&out, blocks * threads * 4); cudaMemset(out, 0, blocks * threads * 4);
    k<<<blocks, threads>>>(slot, out);
    cudaError_t e = cudaDeviceSynchronize();
    unsigned hs[blocks]; unsigned* ho = new unsigned[blocks * threads];
    cudaMemcpy(hs, slot, sizeof hs, cudaMemcpyDeviceToHost);
    cudaMemcpy(ho, out, blocks * threads * 4, cudaMemcpyDeviceToHost);
    bool ok = e == cudaSuccess;
    for (int b = 0; b < blocks && ok; ++b) {
      int winners = 0;
      for (int t = 0; t < threads; ++t) {
        const unsigned v = ho[b * threads + t];
        winners += v == 1;
        ok = ok && (v == 1 || v == 43);
      }
      ok = ok && winners == 1 && hs[b] == 42;
    }
    printf("threads=%d err=%d %s\n", threads, (int)e, ok ? "ok" : "BAD");
    bad += !ok;
    delete[] ho; cudaFree(slot); cudaFree(out);
  }
  printf(bad ? "FAIL\n" : "PASS\n");
  return bad != 0;
}
