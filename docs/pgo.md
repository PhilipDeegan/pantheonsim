# Profile-guided release builds

`scripts/pgo-build.sh` builds the simulator's libraries with GCC's profile-guided optimisation: an instrumented build, a training
run over a manifest of real programs, and an optimised build from the profile. The default build is untouched (PGO stays opt-in:
`VGPU_PGO=generate|use` in `CMakeLists.txt`); the script is what a release job calls. The first measurement (2026-11-08, held-out programs, train/test split) found NO release-wide gain (geometric mean 0.993x, three large
losses): releases do not ship PGO builds, and this is an experiment tool. Whether they should, and what it buys, is measured in [performance.md, "Round sp39 (PGO)"](performance.md).

```
export PGO_CORP=/path/to/corpus PGO_PY=/path/to/python-with-pytorch   # see "The corpus"
scripts/pgo-build.sh check                    # the training programs are all there?
scripts/pgo-build.sh --build build-pgo all    # generate, train, use (strict)
build-pgo/shim/                               # the optimised libraries; LD_LIBRARY_PATH=build-pgo/shim, or vgpu run
```

## Stages

| stage | what it does |
| --- | --- |
| `generate` | `cmake -DVGPU_PGO=generate`, builds every `vgpu*` shared library and `vgpu_cli` (the release targets; not the tests) |
| `train` | runs `scripts/pgo/train-manifest.txt` against that build (`scripts/pgo/train.py`), writes the profile (`<build>/pgo/*.gcda`) and its fingerprint (`<build>/pgo/PROFILE-INFO`) |
| `use` | checks the fingerprint, then `cmake -DVGPU_PGO=use -DVGPU_PGO_STRICT=ON` and builds the same targets |
| `verify` | only the fingerprint check |
| `check` | only the manifest and the existence of the training programs |
| `merge OUT DIR...` | `gcov-tool merge` of profile directories trained in pieces (several machines) |
| `all` | check, generate, train, use |

Options: `--build DIR` (default `build-pgo`), `--corp`, `--py`, `--jobs`, `--train-jobs`, `--cmake-arg ARG` (repeatable, e.g.
`-DCUDA_ABI_INCLUDE=...` on a host without the CUDA toolkit), `--targets`, `--lock FILE` (a `flock` around every build and every
training run, for shared machines), `--min-ok`, `--min-profiled`, `--allow-stale`. `scripts/pgo-build.sh --help` prints them.

Everything happens in ONE build directory: GCC names a profile file after the object file's path
(`#home#me#src#build-pgo#CMakeFiles#vgpu.dir#src#core#x.cpp.gcda`), so the optimised build only finds its profile when it compiles
the same object paths as the instrumented one. The `use` stage reconfigures that directory and rebuilds every object.

## The corpus

Training needs programs built beforehand: nothing at training time needs `nvcc`, `hipcc` or a GPU. The manifest names them with
`@CORP@` (the corpus directory), `@BUILD@`, `@SRC@`, `@PY@`:

| part | where it comes from |
| --- | --- |
| `polybench/bin`, `rodinia/bin`, `hec/src`, `samples` | `tools/bench/real/pack_corpora.sh DIR` on a machine with the Phase 0 corpus built (`~/.cache/{polybenchGpu,gpu-rodinia,HeCBench,cuda-samples}`, sm_86, `-cudart shared`) |
| `amd/bin` | `scripts/pgo/build-amd-corpus.sh DIR`: hipify + `hipcc --offload-arch=gfx942` of the same sources (needs ROCm once; the binaries are what training runs) |
| `pbin` | the pantheon soak workloads, built once with `make PLATFORM=CUDA` in a pantheon checkout (needs nvcc once), copied here |
| PyTorch | a venv with PyTorch for CUDA 13 (`PGO_PY`); training runs `scripts/pgo/run_sweep.py`, a wrapper of `nvidia/tests/pytorch/sweep.py` (see "Traps") |

## The manifests, the held-out set and the weighting

`scripts/pgo/train-manifest.txt` (what the profile is made from) and `scripts/pgo/test-manifest.txt` (programs the profile NEVER
sees, what it is judged by) share one format (documented in `scripts/pgo/manifest.py`): id, engine (`sass`, `ptx`, `amd`, `raw`),
threads, repetitions, time cap, working directory, environment, command. The CI test `lint_pgo_pipeline` fails if a program is in
both.

* Training: NVIDIA programs on the SASS engine (and a subset on the PTX interpreter and at four threads), AMD programs on the AMD
  engine, 14 pantheon workloads, and eleven PyTorch sweep groups in one process.
* Held out: 15 NVIDIA programs (7 of them also on the PTX engine), 7 AMD programs, 12 pantheon workloads, the PyTorch group `nn`,
  5 tiny start-up programs. They include memory-heavy, ALU-heavy, divergent, tiny-launch, shared-memory, atomics and tensor-core
  programs.
* Weighting: a program's repetitions are `round(25 M warp-instructions / its own count)`, between 1 and 8, so that a program of
  4 M instructions runs five times and one of 60 M runs once; the pantheon workloads, which run for a duration, run 3 s each;
  programs that are host-bound (their own CPU code dominates) or longer than 8 s are left out.
* A run that exceeds its cap is killed and counted as CAPPED: an instrumented process killed from outside writes no profile.
  Training fails (exit 1) when fewer than `--min-ok` (0.9) of the entries finished; every failed, capped or missing program is
  listed.

## Cost and hardware

Measured on server2 (Ryzen 9 9950X, 32 threads, GCC 13.3, `-j3`, `nice -n 19`): instrumented build 5 min, training 24 min (122 entries, 395 runs, one process at a time, 179 profile files, 6.9 MB), optimised build 4 min: about 35 minutes of one shared machine, plus the one-off corpus builds. The instrumented libraries run several times slower than plain ones; the training programs are sized for that (a few seconds each, instrumented).

No GPU is needed at any stage. Training runs one process at a time by default (`--train-jobs`); the instrumented libraries
append to the same `.gcda` files under a file lock, so entries may run side by side.

## How often to regenerate

At every release, from the tagged source, in the release job; never commit the `.gcda` files. A profile is valid for ONE compiler
version and ONE revision of every source file it was taken from: any later change to a function's control flow makes that
function's profile a mismatch. Regenerating also when the manifest changes (new programs, new engines) keeps the profile
representative: the PGO win is exactly as large as the training mix is like what users run (the measurements show a profile of
synthetic kernels making PyTorch programs slower).

## Compiler coupling and the stale-profile guard

`.gcda` files are tied to the GCC that wrote them: the file format carries a version (it changes between major releases, so a
profile from GCC 12 does not read with 13), and the per-function control-flow checksums can change between minor releases too.
The pipeline therefore treats any difference in the compiler's full version as stale. `VGPU_PGO` is GCC-only (Clang's
`-fprofile-instr-use` needs `llvm-profdata`; not wired). Three layers make a stale profile fail loudly instead of giving a partial win:

1. `PROFILE-INFO` (written by `train`) records the compiler (`-dumpfullversion`, `-dumpmachine`), a digest of the sources the
   libraries are built from (C/C++ sources, headers, device profiles, CMake files; not tests, docs or scripts) and the build
   directory. The `use` stage compares all three and stops with exit status 3 and the reason; `--allow-stale` skips this one
   layer only.
2. `VGPU_PGO_STRICT` (CMake option, default ON, only meaningful with `VGPU_PGO=use`) compiles with `-Werror=coverage-mismatch`:
   a function whose control flow differs from the profiled one is a compile error. The old setting, `-Wno-coverage-mismatch`,
   compiled such a function without its profile and said nothing, so a profile one commit old gave a partial win with no warning
   (`lint_pgo_pipeline` demonstrates both). The script also greps the build log for the mismatch messages (exit 4).
3. A translation unit with NO profile file is silent under `-Wno-missing-profile` (tests and tools that training does not run
   have none, which is expected), and so is one whose object path differs from the trained one (the profile is then not found).
   After the build the script counts the release translation units that have profile data and fails below `--min-profiled`
   (default 0.5), which catches a moved build directory and a training run that loaded few libraries.

## Traps found while building this

* `nvidia/tests/pytorch/sweep.py` ends in `os._exit()`, which skips every library destructor, and GCC's profile runtime writes
  its files in one: an instrumented PyTorch run wrote NO profile for cuBLAS, cuDNN, cuFFT, cuSOLVER or any library PyTorch loaded
  (89 profile files from the first training run instead of 165 from the PyTorch group alone). `scripts/pgo/run_sweep.py` runs the
  sweep with libc's `exit()` in place of `os._exit`; `lint_pgo_pipeline` checks that difference on a tiny shared library.
* The profile file is named after the object file given to `-o`, so building the same source to another output name finds no
  profile (and `-Wno-missing-profile` hides it): one build directory for all three stages.
* The first run of a cubin on the SASS engine writes a small cache file named after the library's GNU build id
  (`~/.cache/pantheonsim/sass-ok-v1-<id>`): instrumented, plain and optimised builds each have their own, so a timing
  comparison needs a warm-up run per build (the A/B tool does one).

* GCC optimises for size every function the profile never executed. `-DCMAKE_CXX_FLAGS=-fprofile-partial-training` (given to the
  `use` stage with `--cmake-arg`) turns that off; it repaired two of the five large losses of the first measurement and none of the
  shared-memory ones, and it made `test_amd_gcn_same_bits` fail (one `v_mac_f32` result differs): do not use it (performance.md,
  "Round sp39").
* A PGO build fails `test_amd_matrix_kernels_layout`: `-fprofile-use` drops the 64-byte alignment the tree gives its AVX2 tile kernels.
  Run `ctest` on a PGO build before trusting it for anything beyond the held-out timings.

## CI

* `lint_pgo_pipeline` (ctest, `tests/lint/check_pgo_pipeline.sh`; seconds, needs gcc and `gcov-tool`, exit 77 = skipped without):
  the manifests parse and are disjoint; the training runner repeats, caps and classifies entries; the fingerprint rejects a changed
  source, compiler or build path; gcc's generate / merge / use flow with the strict flags rejects a source changed after training;
  `os._exit` versus `exit()` and the profile.
