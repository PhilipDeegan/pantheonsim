# Status and limitations

What PantheonSim runs today, and what it does not. The [README](../README.md) has the short version. [TODO.md](../TODO.md) is the long list of gaps and the history behind this page; where the two disagree, the code and the tests win, so please report it.

The engine is shared between vendors. NVIDIA and CUDA code is in [`nvidia/`](../nvidia/README.md), AMD and ROCm code in [`amd/`](../amd/README.md). [ARCHITECTURE.md](../ARCHITECTURE.md) has the layout.

## Limitations

These come first because they decide whether the simulator fits your use.

**No timing.** It models what a GPU computes, not how long it takes. Performance counters that need a timing model (`%pm0` to `%pm7`, stall reasons, cache hit rates) are not provided. `%clock`, `%clock64` and `%globaltimer` return a deterministic count of the block's instructions, not time. A speed regression or a timing-dependent bug needs a physical GPU. Execution is an interpreter on CPU cores, so runs are much slower than a GPU.

**Synchronous execution.** A kernel launch runs to completion inside the launch call, and streams are synchronous. This has known consequences:

- `cudaLaunchHostFunc` and `cuLaunchHostFunc` run the host function inline. The call blocks, `cudaStreamQuery` and event queries report complete, and a host function that waits for something the caller does later hangs ([#487](https://github.com/pantheongpu/pantheonsim/issues/487)).
- A wait on a false condition cannot be modelled on a synchronous stream, so waiting stream operations are limited ([TODO.md](../TODO.md), gap register).

**A known wrong result.** `shfl.sync` returns 0 for a lane of its member mask that is still in a long loop while the other lanes have reached the shuffle ([#488](https://github.com/pantheongpu/pantheonsim/issues/488)). The simulator does not report this. Other gaps (unsupported instructions or API calls) fail with an error that names the instruction, PTX line, kernel and profile.

**Shared cudart only.** CUDA programs must link the shared runtime (`nvcc -cudart shared`) so the loader can substitute the simulator's `libcudart.so.13`. Hosting a statically linked cudart needs NVIDIA's undocumented driver export tables; this was investigated and is blocked (see [TODO.md](../TODO.md), "Next milestones"). CuPy links cudart statically and does not run ([nvidia/docs/jit.md](../nvidia/docs/jit.md)).

**Not supported:**

- OptiX (ray tracing) is a separate NVIDIA subsystem and is out of scope.
- NVIDIA's Nsight Systems and Nsight Compute. `vgpu ncu` is this project's own `ncu`; `nsys` runs but collects no CUDA data ([nvidia/docs/cupti.md](../nvidia/docs/cupti.md)).
- NVDEC decodes Motion JPEG, H.264, HEVC and MPEG-2 only. MPEG-1, MPEG-4, VC-1, VP8, VP9 and AV1 report unsupported.
- The `wmma` fragment layout is the simulator's own, because PTX leaves it unspecified ([ARCHITECTURE.md](../ARCHITECTURE.md), D8). Kernels that load, multiply and store fragments through `wmma` get the same results as hardware; a kernel that depends on NVIDIA's element distribution may differ.
- Divergence uses min-PC reconvergence, not full immediate-post-dominator handling. Irreducible control flow is not handled, and no test pins that ([TODO.md](../TODO.md), "Partially implemented").

## How accurate it is

- **Profiles.** 48 in total: 38 NVIDIA and 10 AMD (`vgpu list-gpus`; `nvidia/profiles/` and `amd/profiles/`). A profile marked `verified: true` was read from a physical card. That is 14 NVIDIA profiles and 3 AMD profiles; the other 31 are built from public documents and nothing in them was measured. Each profile's header says where its values came from.
  - Verified NVIDIA: T4, A10, A10G, A100 (two profiles, `a100` and `a100-sxm4-40gb`), H100 and H100 PCIe, L4, L40S, RTX 3060, RTX 3080 Ti, GH200, RTX PRO 6000 Server Edition and RTX PRO 4500. Each reproduces the 512 values of the conformance checks run against that card.
  - Verified AMD: MI325X, RX 6800, RX 6700 XT.
  - Unverified profiles include H200, B200, B300, GB200, RTX 4090 and RTX 5090 and the other AMD GPUs (MI250X, MI300X, MI350X, MI455X, RX 6900 XT, RX 7900 XTX, RX 9070 XT). Rubin (`vr200`) and Jetson Thor carry placeholder PCI ids because NVIDIA has published none.
  - Full tables: [nvidia/docs/profiles.md](../nvidia/docs/profiles.md) and [amd/README.md](../amd/README.md).
- **Libraries.** The NVIDIA library shims are compared with NVIDIA's own libraries on a physical GPU for the cases listed in [nvidia/docs/libraries.md](../nvidia/docs/libraries.md) ("Verified against real hardware"). Libraries with no NVIDIA binary to compare against, or that are written from public documentation, say so there.
- **Tests.** `make test` runs the suite; the CI jobs are listed in [docs/ci.md](ci.md). Tests that compile CUDA or HIP skip when `nvcc` or `hipcc` is missing.
- **Workloads.** The pantheon stress workloads run on the simulator with `--verify` and are described in [docs/pantheon-workloads.md](pantheon-workloads.md). `rt_virus` needs OptiX and is excluded.

## What runs

### NVIDIA and CUDA

| Area | What works | Details |
| --- | --- | --- |
| Compilers | Programs built with `nvcc` from CUDA 12 or 13 (`-cudart shared`). Kernels run from SASS when the binary has it (sm_75 to sm_120a), else from PTX. `VGPU_SASS=0` forces PTX | [sass.md](../nvidia/docs/sass.md) |
| PTX | A large subset of the ISA, including `wmma`, `mma.sync`, `ldmatrix`, `cp.async`, bf16, textures, surfaces, grid sync, named barriers, Hopper `wgmma`, TMA, cluster barriers and distributed shared memory. Checked against CuTe and CUTLASS Hopper unit tests | [ARCHITECTURE.md](../ARCHITECTURE.md), [TODO.md](../TODO.md) |
| Driver and runtime API | `libvgpucuda.so` and `libvgpucudart` (presented as `libcudart.so.13`): contexts, memory, modules, streams, events, graphs, memory pools, `cuMem*` virtual memory, multi-GPU with peer copies. Every function declared in `cuda.h` and `cuda_runtime_api.h` is exported; ones not implemented return a "not supported" status | [docs/driver-api.md](driver-api.md), [nvidia/docs/devices.md](../nvidia/docs/devices.md) |
| Virtual memory | Sparse backing, so a simulated H200 reports 141 GB on a smaller host. Past `ram_mb` memory spills to disk. A host-memory guard turns a host out-of-memory into a device out-of-memory error. Out-of-bounds, use-after-free, double-free and misaligned accesses are reported | [docs/config.md](config.md) |
| Libraries | cuBLAS, cuBLASLt, cuDNN, cuFFT, cuRAND, cuSPARSE, cuSPARSELt, cuSOLVER, cuTENSOR, cuTensorNet, cuStateVec, cuDSS (0.8 and 1.0), NCCL, NVRTC, NPP, nvJPEG, nvJitLink, nvFatbin, cuFile (compatibility mode), nvCOMP, NVSHMEM, under their real sonames | [libraries.md](../nvidia/docs/libraries.md) |
| Frameworks | PyTorch's official CUDA 13 build runs on the simulated NVIDIA GPUs; its 12 whole-model checks (GPT, ResNet, LSTM, U-Net, ViT, MoE, DLRM and others) were run on seven profiles (T4, A100, L4, RTX 3060, H100, B200, RTX 5090) and compared with the CPU. Numba and Triton run unmodified | [pytorch.md](../nvidia/docs/pytorch.md), [jit.md](../nvidia/docs/jit.md) |
| Discovery | NVML, `nvidia-smi` and `lspci` output for the simulated GPUs | [docs/telemetry.md](telemetry.md) |
| Registers | PCI configuration space from a register database, `vgpu regs list/read/write/dump/log`, a C API (`vgpu_regs.h`) | [docs/registers.md](registers.md) |
| Profiling | CUPTI's Activity and Callback APIs under the real soname; `torch.profiler` collects a GPU trace; unmodified `nvprof` works on the T4 profile. Event and Profiling metrics APIs are absent | [cupti.md](../nvidia/docs/cupti.md) |
| Video | NVENC (`libnvidia-encode.so.1`) writes H.264 and HEVC streams that ffmpeg decodes. NVDEC (`libnvcuvid.so.1`) decodes Motion JPEG, H.264 (Baseline, Main, High), HEVC (Main, Main 10, 4:2:0 range extensions) and MPEG-2 (Main) | [libraries.md](../nvidia/docs/libraries.md) |

### AMD and ROCm

| Area | What works | Details |
| --- | --- | --- |
| Programs | `hipcc`-built programs from ROCm 6.4, 7.0, 7.1 and 7.2, through the simulator's own HIP and HSA runtimes | [amd/README.md](../amd/README.md) |
| Architectures | gfx90a (MI250X), gfx942 (MI300X, MI325X), gfx950 (MI350X), gfx1250 (MI455X), gfx1030 and gfx1031 (RX 6900 XT, RX 6800, RX 6700 XT), gfx1100 (RX 7900 XTX), gfx1201 (RX 9070 XT). gfx942 code is checked instruction by instruction against ROCm's `llvm-objdump` | [amd/README.md](../amd/README.md) |
| Libraries and tools | rocBLAS, hipBLASLt, MIOpen, rocPRIM/hipCUB, rocRAND, rocFFT, rocSPARSE, rocSOLVER, RCCL, PyTorch for ROCm; `rocprofv3`, `rocm-smi` and `amd-smi`. rocBLAS passes 162,807 float and double tests of its own quick suite | [amd/README.md](../amd/README.md), [rocprofiler.md](../amd/docs/rocprofiler.md) |

The `vgpu` command-line tool (`list-gpus`, `info`, `demo`, `run`, `shell`, `config`, `test --matrix`, `ncu`, `counters`, `debug`, `fault`, `regs`, `smi`, `serve`) is documented in [cli.md](cli.md). Random and adversarial warp scheduling and shared-memory race detection are in [config.md](config.md) (`scheduler`, `race`).

## Open work

[TODO.md](../TODO.md) holds the full gap register. The largest items:

1. Static-cudart hosting (blocked, see above).
2. Asynchronous streams and host functions, which would remove the synchronous-engine limits ([#487](https://github.com/pantheongpu/pantheonsim/issues/487)).
3. The `shfl.sync` wait ([#488](https://github.com/pantheongpu/pantheonsim/issues/488)).
4. Reading the remaining profiles from physical cards, and a differential-testing database against real GPUs.

Some entries in TODO.md predate later work (for example, it still lists the driver graph API as absent although `nvidia/src/driver_graph.cpp` implements it); treat its gap register as a lead to check, not a verified list.
