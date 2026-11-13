# Real-program A/B (tools/bench/real)

`tools/bench` and `tools/bench-amd` time small synthetic kernels. This
directory times whole prebuilt CUDA programs on the simulator, with two builds
side by side, so a speed-up can be checked against real code and the next
target found.

Corpora (all built for sm_86 with `-cudart shared`, simulator side only; no
GPU card is used):

| corpus | programs | source |
|---|---|---|
| polybench | 20 PolyBench/GPU kernels, MINI dataset | `~/.cache/polybenchGpu/bin` |
| rodinia | 12 Rodinia apps, sizes scaled up from the card/sim runner so they run for seconds | `~/.cache/gpu-rodinia/bin` |
| hecbench | the 34 reduced HeCBench benchmarks (`reduced.txt`) | `~/.cache/HeCBench` |
| samples | the CUDA Samples v13.0 that pass on the simulator (`sim_main.txt`) | `~/.cache/cuda-samples` |

## Use

```
tools/bench/real/pack_corpora.sh /tmp/corp          # on the dev machine: copies files only
tar -C /tmp/corp -cf - . | ssh server3 'mkdir -p /tmp/corp && tar -C /tmp/corp -xf -'
# on the timing box (both builds need <build>/shim):
python3 tools/bench/real/real.py --before /tmp/base0/build --after /tmp/sp5-fin3/build \
    --corp /tmp/corp --cpu 0 --runs 5 --threads 1 --timeout 150 --json t1.json
python3 tools/bench/real/real.py ... --threads default      # second table
python3 tools/bench/real/profile.py --build /tmp/sp5-fin3/build --corp /tmp/corp \
    --programs polybench/LU,hecbench/bh-cuda --cpu 14     # where the time goes
```

* Each run is `taskset -c <cpu> <program>` with `VGPU_QUIET=1`,
  `VGPU_GPU=nvidia/rtx3060`, `LD_LIBRARY_PATH=<build>/shim` and (unless
  `--threads default`) `VGPU_THREADS=1`. BEFORE and AFTER alternate, one run at
  a time; the table shows median wall seconds and median user CPU seconds.
* The harness waits while the 1-minute load average is above `--max-load`
  (default 8). Timing under load is not usable: say so in a report if the box
  never calms down.
* Programs slower than `--long-secs` (30) in their first pair get only
  `--long-runs` (3) runs. `--skip` drops known timeouts; a run longer than
  `--timeout` is killed and shown as TIMEOUT.
* "outputs" is `same` when the normalised stdout/stderr (lines carrying
  timings, rates or the platform banner are dropped) and, for Rodinia, the
  files the program wrote hash equal in every run of both builds. "checks"
  lists the PASS/FAIL/Non-Matching/RMSE lines the program printed. A DIFFER
  row is a correctness question before it is a speed one.
* Wall time includes the host-side part of the program (CPU reference
  checks in PolyBench and many HeCBench programs): `profile.py` reports the
  share of cycles spent in the program's own binary, which no simulator
  change can remove.
* Very short programs (< 0.3 s) time process start-up, not simulation.

## Short programs: where start-up goes (`short.py`)

`real.py` scales the Rodinia sizes up so that it times simulation. `short.py`
goes the other way: the same corpora, with Rodinia at the sizes of the card/sim
runner (tens of milliseconds), to see what a program that finishes well under a
second, as most CI programs do, spends its wall time on.

```
python3 tools/bench/real/short.py survey --build B --corp C                   # wall time of every candidate
python3 tools/bench/real/short.py split  --build B --corp C --programs samples/vectorAdd,rodinia/bfs \
    --reps 10 --sudo [--env VGPU_SASS=0] [--top 8] [--keep DIR] [--json F]    # the phases of each, by perf
python3 tools/bench/real/short.py ab     --build B --corp C --programs ... --runs 7 \
    --variant off:VGPU_SASS_CACHE=0 --variant 'cold:PRE=rm -f <cache>/sass-ok-v1-*' --variant warm:
python3 -I tools/bench/real/short.py selftest                                  # the phase rules (CMake: lint_short_split_selftest)
```

* `split` records each program `--reps` times with `perf record -F 20000 -g -e
  cycles` (user and kernel, one CPU) and puts every sample of the program's
  process in one phase by its call chain: `loader` (exec and the dynamic
  loader), `device-init` (profile, device, telemetry), `module-load` (fatbin
  and cubin), `sass-decode` (the check that a cubin runs in full, and decoding
  a code section when first run), `tier-build` (the pre-decoded tier's
  analysis, the JIT), `ptx-parse` (parsing and lowering PTX), `execute`,
  `teardown`, `sim-other` (the simulator's API outside those: allocation,
  memcpy, page faults of device memory), `program-host` (the program's own
  code and libc) and `kernel-other`. The first rule that matches any frame of
  the chain wins, in that order of specificity, so decoding done inside the
  first launch is decoding. A rule looks at a frame's function name only (no
  parameter list, no template arguments). Chains are cut short under the
  tier's handlers; a sample no chain rule placed is `execute` when its leaf is
  one of the executor's leaves.
* perf needs `kernel.perf_event_paranoid` <= 1 or root. `--sudo` runs
  `sudo -n perf record ... -- sudo -n -u <you> -- env -i ... <program>`: perf
  as root, the program as you, no system setting changed.
* `ab` alternates any number of named variants (environment settings, and
  `PRE=<shell command>` run before each run of that variant), medians of
  `--runs`, the same output check as `real.py`, and the geometric mean of the
  first variant over each of the others.
* Set `XDG_CACHE_HOME` to a directory of your own first: the SASS verdict file
  lives there, and `cold` deletes it.
