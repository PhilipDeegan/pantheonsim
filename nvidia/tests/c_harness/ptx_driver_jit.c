/*
 * ptx_driver_jit.c -- PTX the driver accepts or refuses, from a plain C program, checked against an
 * RTX 3060 (driver 596, CUDA 13.0, PTX ISA 9.3).
 *
 *  - ld.weak / st.weak are PTX's explicit spelling of the default memory semantics (GiMMiK's kernels
 *    write every load and store that way) and run like the plain forms.
 *  - A .reg range may declare far more than 65,536 registers: ptxas counts what the kernel uses
 *    (nvcc's unrolled BN254 field arithmetic declares 125,511).
 *  - cuModuleLoadDataEx fills CU_JIT_ERROR_LOG_BUFFER (and the size slot) when a load fails. Malformed
 *    PTX is CUDA_ERROR_INVALID_PTX (218), a .version past 9.3 is CUDA_ERROR_UNSUPPORTED_PTX_VERSION
 *    (222), and text that does not begin with .version is not recognised at all (200).
 *  - cuLinkAddData compiles its input as it is added, so the same three failures show there, with the
 *    log, and cuLinkComplete is never reached.
 *  - With an argument naming libnvrtc: the options NVRTC takes for floating-point behaviour
 *    (--ftz, --prec-div, --prec-sqrt, --fmad) are accepted by the built-in compiler.
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Built with -DPTX_JIT_REAL_CUDA against the toolkit's cuda.h, the same program runs on a card: the
 * expectations above are what it printed. */
#ifdef PTX_JIT_REAL_CUDA
#include <cuda.h>
#else
#include "vgpu_cuda.h"

typedef int CUjit_option;
typedef struct CUlinkState_st* CUlinkState;
CUresult cuLinkCreate(unsigned int numOptions, int* options, void** optionValues, CUlinkState* stateOut);
CUresult cuLinkAddData(CUlinkState state, int type, void* data, size_t size, const char* name,
                       unsigned int numOptions, int* options, void** optionValues);
CUresult cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut);
CUresult cuLinkDestroy(CUlinkState state);
#endif

enum { kErrorLog = 5, kErrorLogSize = 6 };
static int failures = 0;

static void expect(int ok, const char* what) {
  printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}

static const char kHdr[] = ".version 8.0\n.target sm_80\n.address_size 64\n";

static CUresult load_ex(const char* ptx, char* log, size_t cap, size_t* written) {
  CUjit_option opts[2] = {kErrorLog, kErrorLogSize};
  void* vals[2] = {log, (void*)cap};
  CUmodule m;
  memset(log, 0, cap);
  CUresult r = cuModuleLoadDataEx(&m, ptx, 2, opts, vals);
  *written = (size_t)vals[1];
  return r;
}

static CUresult link_add(const char* ptx, char* log, size_t cap, size_t* written, int complete) {
  CUjit_option opts[2] = {kErrorLog, kErrorLogSize};
  void* vals[2] = {log, (void*)cap};
  CUlinkState st;
  memset(log, 0, cap);
  if (cuLinkCreate(2, opts, vals, &st) != CUDA_SUCCESS) return (CUresult)-1;
  CUresult r = cuLinkAddData(st, 1 /* PTX */, (void*)ptx, strlen(ptx) + 1, "in.ptx", 0, NULL, NULL);
  if (r == CUDA_SUCCESS && complete) {
    void* cubin;
    size_t sz;
    r = cuLinkComplete(st, &cubin, &sz);
  }
  *written = (size_t)vals[1];
  if (r != CUDA_SUCCESS) cuLinkDestroy(st);  /* a good state is left: the module may use its image */
  return r;
}

/* Runs `k(out)`, which stores 7 at *out, and returns what it stored (0 if anything failed). */
static unsigned long long run_k(const char* ptx) {
  CUmodule m;
  CUfunction f;
  CUdeviceptr d;
  unsigned long long h = 0;
  void* args[1];
  if (cuModuleLoadData(&m, ptx) != CUDA_SUCCESS) return 0;
  if (cuModuleGetFunction(&f, m, "k") != CUDA_SUCCESS) return 0;
  if (cuMemAlloc(&d, 8) != CUDA_SUCCESS) return 0;
  args[0] = &d;
  if (cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, 0, args, 0) != CUDA_SUCCESS) return 0;
  if (cuCtxSynchronize() != CUDA_SUCCESS) return 0;
  cuMemcpyDtoH(&h, d, 8);
  return h;
}

int main(int argc, char** argv) {
  CUdevice dev;
  CUcontext ctx;
  char log[4096], ptx[8192];
  size_t w;
  CUresult r;
  if (cuInit(0) || cuDeviceGet(&dev, 0) || cuDevicePrimaryCtxRetain(&ctx, dev) || cuCtxSetCurrent(ctx)) {
    fprintf(stderr, "FAIL: no device\n");
    return 1;
  }

  /* ld.weak / st.weak */
  snprintf(ptx, sizeof ptx,
           "%s.visible .entry k(.param .u64 p)\n{\n  .reg .b64 %%rd<2>;\n  .reg .f64 %%fd<2>;\n"
           "  ld.param.u64 %%rd0, [p];\n  mov.f64 %%fd0, 0d401C000000000000;\n"
           "  st.weak.global.f64 [%%rd0], %%fd0;\n  ld.weak.global.cg.f64 %%fd1, [%%rd0];\n"
           "  st.weak.global.f64 [%%rd0], %%fd1;\n  ret;\n}\n",
           kHdr);
  {
    unsigned long long bits = run_k(ptx);
    double v;
    memcpy(&v, &bits, 8);
    expect(v == 7.0, "ld.weak / st.weak load, store and run");
  }

  /* 70,000 declared registers, two used */
  snprintf(ptx, sizeof ptx,
           "%s.visible .entry k(.param .u64 p)\n{\n  .reg .b64 %%r<70000>;\n  .reg .b64 %%a;\n"
           "  ld.param.u64 %%a, [p];\n  mov.u64 %%r1, 7;\n  st.global.u64 [%%a], %%r1;\n  ret;\n}\n",
           kHdr);
  expect(run_k(ptx) == 7, "70,000 declared registers load and run");

  /* a failed load fills the error log */
  r = load_ex(".version 7.0\n.target sm_50\n.address_size 64\nthis is not valid ptx\n", log, sizeof log, &w);
  expect(r == 218, "malformed PTX is CUDA_ERROR_INVALID_PTX (218)");
  expect(strlen(log) > 0 && w == strlen(log), "malformed PTX leaves an error log, and its length in the size slot");
  expect(strstr(log, "line 4") != NULL, "the log names the line");
  r = load_ex(kHdr, log, sizeof log, &w);
  expect(r == CUDA_SUCCESS && log[0] == 0 && w == 0, "a good load leaves the log empty");
  r = load_ex(".version 99.9\n.target sm_80\n.address_size 64\n", log, sizeof log, &w);
  expect(r == 222, "a .version past 9.3 is CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222)");
  expect(strstr(log, "Unsupported .version 99.9") != NULL, "the log says which version");
  r = load_ex(".version 9.4\n.target sm_80\n.address_size 64\n", log, sizeof log, &w);
  expect(r == 222, ".version 9.4 is refused");
  r = load_ex(".version 9.3\n.target sm_80\n.address_size 64\n", log, sizeof log, &w);
  expect(r == CUDA_SUCCESS, ".version 9.3 is accepted");
  r = load_ex("garbage", log, sizeof log, &w);
  expect(r == 200, "text that is not PTX is CUDA_ERROR_INVALID_IMAGE (200)");
  r = load_ex("// a comment\n  .version 8.0\n.target sm_80\n.address_size 64\n", log, sizeof log, &w);
  expect(r == CUDA_SUCCESS, "comments and white space may come before .version");
  r = load_ex(".target sm_80\n.version 8.0\n.address_size 64\n", log, sizeof log, &w);
  expect(r == 200, ".target before .version is not recognised");
  {
    char small[16];
    r = load_ex(".version 7.0\n.target sm_50\n.address_size 64\nthis is not valid ptx\n", small, sizeof small, &w);
    expect(r == 218 && strlen(small) == 15 && w == 15, "the log is cut to the buffer, with its terminator");
  }

  /* the linker compiles an input as it is added */
  r = link_add(".version 99.9\ngarbage", log, sizeof log, &w, 1);
  expect(r == 222, "cuLinkAddData: .version 99.9 is 222, not success");
  expect(strstr(log, "Unsupported .version 99.9") != NULL && w == strlen(log), "cuLinkAddData: the log says why");
  r = link_add(".version 7.0\n.target sm_50\n.address_size 64\nthis is not valid ptx\n", log, sizeof log, &w, 1);
  expect(r == 218 && strlen(log) > 0, "cuLinkAddData: malformed PTX is 218 with a log");
  r = link_add("garbage", log, sizeof log, &w, 1);
  expect(r == 200, "cuLinkAddData: text that is not PTX is 200");
  r = link_add(kHdr, log, sizeof log, &w, 1);
  expect(r == CUDA_SUCCESS && log[0] == 0, "cuLinkAddData and cuLinkComplete take good PTX");

  /* NVRTC's floating-point options */
  if (argc > 1) {
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    typedef int (*create_t)(void**, const char*, const char*, int, const char* const*, const char* const*);
    typedef int (*compile_t)(void*, int, const char* const*);
    create_t create = h ? (create_t)dlsym(h, "nvrtcCreateProgram") : NULL;
    compile_t compile = h ? (compile_t)dlsym(h, "nvrtcCompileProgram") : NULL;
    typedef int (*destroy_t)(void**);
    destroy_t destroy = h ? (destroy_t)dlsym(h, "nvrtcDestroyProgram") : NULL;
    if (!create || !compile) {
      expect(0, "libnvrtc loads");
    } else {
      static const char* const kOpts[] = {"--ftz=true", "--ftz=false", "--prec-div=false", "--prec-sqrt=true", "--fmad=true"};
      unsigned i;
      for (i = 0; i < sizeof kOpts / sizeof *kOpts; ++i) {
        void* prog = NULL;
        const char* opts[2] = {"--gpu-architecture=compute_80", kOpts[i]};
        char what[80];
        int rc;
        create(&prog, "extern \"C\" __global__ void k(float* x) { x[0] = 1.f; }", "k.cu", 0, NULL, NULL);
        rc = compile(prog, 2, opts);
        snprintf(what, sizeof what, "NVRTC takes %s (result %d is not NVRTC_ERROR_INVALID_OPTION)", kOpts[i], rc);
        expect(rc != 5, what);
        if (destroy && prog) destroy(&prog);
      }
    }
  }
  if (failures) {
    printf("%d FAILED\n", failures);
    return 1;
  }
  printf("PASS ptx_driver_jit\n");
  return 0;
}
