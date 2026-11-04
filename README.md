# PantheonSim

A CPU-only simulator of NVIDIA and AMD GPUs. It provides simulated driver, runtime and library files (`libcuda`, `libcudart`, the HIP runtime, `nvidia-smi` and others), so a program built with the real `nvcc` or `hipcc` can run on a machine with no GPU. A profile per GPU model (a T4, an H100, an MI300X and 45 others) sets what the program sees.

It checks what a program computes, not how fast a GPU would run it. There is no timing model.

[![CI](https://github.com/pantheongpu/pantheonsim/actions/workflows/ci.yml/badge.svg)](https://github.com/pantheongpu/pantheonsim/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/pantheongpu/pantheonsim)](https://github.com/pantheongpu/pantheonsim/releases)
[![License](https://img.shields.io/github/license/pantheongpu/pantheonsim)](LICENSE)

## Limits

- No timing. Timings from a run mean nothing, and `%clock` and `%globaltimer` count instructions, not time. Run on a physical GPU before a release if speed or real hardware behaviour matters.
- The simulator executes kernels on CPU cores, so it is much slower than a GPU.
- A CUDA program must link the shared runtime (`nvcc -cudart shared`). Programs that link cudart statically, such as CuPy, do not run.
- Launches and streams are synchronous: a kernel runs to completion inside the launch call, and `cudaLaunchHostFunc` runs the host function inline ([#487](https://github.com/pantheongpu/pantheonsim/issues/487)). Code that depends on work being in flight can hang or see work as already done.
- Open bug: `shfl.sync` returns 0 for a lane of its mask that is still in a long loop ([#488](https://github.com/pantheongpu/pantheonsim/issues/488)).
- Not supported: OptiX, and NVIDIA's own Nsight Systems and Nsight Compute (`vgpu ncu` is this project's replacement for `ncu`).
- 31 of the 48 GPU profiles come from public documents only and were not measured on a card (marked unverified in the profile).
An unsupported instruction or API fails with an error that names it. [Status and limits](docs/status.md) has the rest.

## Quickstart with Docker

No GPU, driver or `--gpus` flag is needed:

```bash
docker run --rm ghcr.io/pantheongpu/pantheonsim:latest vgpu list-gpus
docker run --rm -e VGPU_GPU=nvidia/h100 -e VGPU_DEVICE_COUNT=2 ghcr.io/pantheongpu/pantheonsim:latest nvidia-smi
```

The second command prints `nvidia-smi` output for two simulated H100s. The image has no `nvcc` or `hipcc`; [docs/docker.md](docs/docker.md) shows how to build a program inside a container.

## Build from source

```bash
make     # installs cmake and a C++ compiler if missing (asks first), then builds
./run    # starts a simulated GPU machine and takes you inside it; type exit to leave
```

Requirements: Linux (WSL works), CMake and a C++20 compiler. Run both commands from the repository root. `make doctor` lists what is missing, `make deps` installs it, and `make build` builds without installing anything. The build takes a few minutes and uses half of the CPUs (`make JOBS=8` to choose). The simulator also uses half of the CPUs; set `threads: N` in the settings file, or `make run THREADS=N`, to change that.

`./run` reads `pantheonsim.yaml` and writes one on the first run. `./run pantheonsim.example.yaml` runs the sample file as it is (one simulated H100); copy it to `pantheonsim.yaml` to make it your default.

Other targets:
```bash
make demo                           # run a built-in vectorAdd kernel on a simulated H100; ends with PASS
make run APP=./my_app ARGS="..."    # run one program on the simulated GPU, without the machine shell
make test                           # build and run the test suite
make help                           # every target and setting
```

Inside the machine, `nvidia-smi` shows the GPUs. Compilers, `curl`, `git` and Python come from your own machine. Use a virtual environment for Python packages (`python3 -m venv ~/v && ~/v/bin/pip install ...`). To build your own programs you need the vendor compiler installed; see [docs/toolchains.md](docs/toolchains.md).

## Choose the GPU

`make list-gpus` (or `build/vgpu list-gpus`) prints the 48 ids you can use: 38 NVIDIA (T4 to B300) and 10 AMD (RX 6700 XT to MI455X). Set one with `gpu: nvidia/t4` in the settings file, or `make run GPU=nvidia/t4`. `build/vgpu info --gpu nvidia/t4` shows one profile. 14 NVIDIA and 3 AMD profiles were read from physical cards; the rest come from public documents. [Status and limits](docs/status.md) lists which.

Put settings in a file such as `my.yaml` and run `./run my.yaml`:

```yaml
gpu: nvidia/h100     # which GPU (build/vgpu list-gpus shows them all)
count: 8             # how many
strict: true         # refuse undefined behaviour, such as dividing by zero
```

All settings ([Settings](docs/config.md) describes each; `build/vgpu config keys` prints them):

| Key | Values | Default | What it does |
| --- | --- | --- | --- |
| `gpu` | an id from `list-gpus`, such as `nvidia/h100` | `nvidia/h100` | The GPU model to present |
| `count` | 1 to 16 | 1 | How many identical devices |
| `vram_mb` | MiB, at least 1 | the profile's own | Memory per device |
| `cuda_version` | `major.minor`, such as `13.0` | this build's | The CUDA version `nvidia-smi` and NVML report |
| `driver_version` | such as `580.65.06` | this build's | The NVIDIA driver version reported |
| `rocm_version` | such as `6.4.0` | `6.2.0` in the machine shell | The ROCm version `amd-smi version` reports |
| `os` | `ubuntu`, `rocky` or `debian`, with a version (`ubuntu:24.04`) | `ubuntu:22.04` | The operating system the machine presents |
| `ram_mb` | MiB, at least 1 | unlimited | Host RAM device memory may use before it spills to `memory_dir` |
| `memory_dir` | a directory | never spill | Where device memory past `ram_mb` goes (runs at disk speed) |
| `host_reserve_mb` | MiB, 0 or more | 6% of the limit, 1 to 4 GiB | Memory the guard leaves to the rest of your machine |
| `host_guard` | `true` or `false` | `true` | `false` turns the host-memory guard off |
| `threads` | 1 to 65536 | one per core (half under `make run` and `./run`) | Host threads that run blocks |
| `max_steps` | 0 or more | built in | Per-thread instruction budget against a kernel that never ends; `0` removes it |
| `sass` | `auto`, `off` or `only` | `auto` | The engine for NVIDIA kernels: SASS when the binary has it, or force PTX, or insist on SASS |
| `simd` | `auto`, `base`, `avx2` or `off` | `auto` | Caps the vector instruction set; `off` is scalar code |
| `scheduler` | `deterministic`, `random` or `adversarial` | `deterministic` | Warp scheduling order, to find order-dependent bugs (set `sass: off` with the last two) |
| `scheduler_seed` | 0 or more | 0 | Seed for `random` and `adversarial`; the same seed replays the same run |
| `strict` | `true` or `false` | `false` | Refuse undefined behaviour, such as integer division by zero |
| `race` | `off`, `on` or `all` | `off` | Report unordered shared-memory access between warps |
| `counters` | `true` or `false` | `false` | Print exact per-launch performance counters |
| `quiet` | `true` or `false` | `false` | Silence every simulator message on stderr |
| `trace` | `true` or `false` | `false` | Log every runtime and driver entry point |

Other ways to run it: in a browser at [pantheonsim.com/play](https://pantheonsim.com/play/); in CI with `pantheongpu/setup-pantheonsim@v0` ([docs/github-action.md](docs/github-action.md)) or, on other CI systems, `scripts/ci-setup.sh` ([docs/ci-other-platforms.md](docs/ci-other-platforms.md)).

## Documentation

[Getting started](docs/getting-started.md) | [Settings](docs/config.md) | [The machine](docs/machine-simulator.md) | [Compilers](docs/toolchains.md) | [Commands](docs/cli.md) | [Status and limits](docs/status.md) | [Architecture](ARCHITECTURE.md) | [Contributing](CONTRIBUTING.md) | [Code of conduct](CODE_OF_CONDUCT.md) | [Governance](GOVERNANCE.md) | [Security](SECURITY.md)

Questions go in [Discussions](https://github.com/pantheongpu/pantheonsim/discussions), bugs in [issues](https://github.com/pantheongpu/pantheonsim/issues). Apache-2.0 (see LICENSE). CUDA is a trademark of NVIDIA; this project is not affiliated with NVIDIA or AMD.
