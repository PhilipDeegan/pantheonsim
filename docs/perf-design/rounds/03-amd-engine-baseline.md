## The AMD engine

The sections above measure the NVIDIA engines (SASS and PTX). The AMD interpreter
(`amd/src/gcn_exec.cpp`, `gcn_decode.cpp`, `rdna_decode.cpp`) had no baseline. This
section is that baseline and its first profile. **No simulator source was changed to produce
it**: the harness, workloads and profiler are new files under `tools/bench-amd/`, and one
`add_test` in `CMakeLists.txt` wires the counter check. Every number was measured by the
commands shown.

**Measured at** `speedup` 8f9eb524 plus the files of this section; Release build (`-O3
-DNDEBUG`, LTO, GCC 13.3.0, baseline x86-64: no POPCNT, F16C, FMA or AVX2 outside the few
`target(...)` functions); programs built by `hipcc` of ROCm 7.2 (HIP 7.2.26015) for `gfx942`
and `gfx1030`, run on the `amd/mi300x` and `amd/rx6800` profiles.

**Machine**: server2, AMD Ryzen 9 9950X (16 cores, 32 logical CPUs) shared with a Postgres,
a CI runner and other people's builds. Every run was pinned (`taskset`) to CPUs 24-31 (the
SMT siblings of cores 8-15). The 1-minute load average before each workload was 9 to 42
during run A below and 10 to 52 (median 35) during run B; the A/B experiments at the end ran at
load 10 to 22, 10 to 13 and 27 to 46. The numbers compare two builds on one machine; the
spreads say how much the machine took away. On this host read the **best** column.

### The harness

```
tools/bench-amd/bench.sh --build build --cpus 24-31          # time every workload, MI300X and RX 6800
tools/bench-amd/bench.sh --build buildA --build buildB       # alternating A B A B ... with ratio and p
tools/bench-amd/bench.sh --check-counters --build build      # counts (not times) against tools/bench-amd/expected
tools/bench-amd/bench.sh --update-expected --build build     # rewrite expected/amd-hip-<release>.json
tools/bench-amd/profile/profile.sh build-g valu_int mi300x   # sampled profile of one workload
```

`tools/bench-amd/bench.py` is the AMD twin of `tools/bench/bench.py` (it imports that file's
statistics, Mann-Whitney p-value and process runner; the verdict rule is the same).

* The workloads are `kernels/bench_amd.hip`, one self-checking HIP program
  (`bench_amd <workload> <quick|full>`) compiled per target with `hipcc -O2 -ffp-contract=off`
  and run through VirtualGPU's `libamdhip64` like any user program. Each workload compares its
  results with a host reference bit for bit and prints `PASS` only on a match. A timed run that
  fails is recorded, repeated (up to three times) and listed at the end, never dropped. (One
  first-attempt failure was seen once, during the first attempt of run B before this was added: valu_int
  printed MISMATCH with a device phase of 0.4 ms, i.e. the kernel apparently did not run; 160 quick runs
  under heavy CPU load and 25 harness repeats could not reproduce it, and run B as reported had no failures.
  Launch errors are checked now.

  *Explained later (`speedup4-amdcorrect`).* About 28,000 further `valu_int` quick runs on the MI300X, in loops of 4
  to 16 at once on 4 to 8 CPUs of a box at load 35 to 95, at `VGPU_THREADS` 1, 3, 16 and the default, found no
  race: none gave a wrong answer, and the queue, the worker pool, the decode cache and the launch/synchronize
  hand-off, read for a missed wake-up, an early completion or a stale flag, have none. What did fail, in 3 bursts of
  50 to 400 consecutive runs on server2 while other sessions' builds held 28 of its 30 GB and its swap was full, was the
  launch itself: the simulator's host-memory guard (`VGPU_MEMORY_HOST_RESERVE_MB`, default 6% of RAM clamped to 1 to
  4 GiB) refuses device memory while `MemAvailable` is below the reserve (the message, with `VGPU_QUIET` unset:
  "device out of memory: the host has 618 MB of memory left ... below the reserve of 1840 MB"), the first launch of a
  program writes its linked code object into device memory (`place`), and that refusal came back as
  `hipErrorInvalidImage` ("device kernel image is invalid"), blaming the program. A program that does not read
  `hipGetLastError` after a launch then synchronizes (nothing queued), reads the untouched buffer back, and prints
  MISMATCH with the time of a launch that did nothing: the 0.4 ms. Reproduced on demand with the guard's reserve set
above the host's memory and the launch check taken out of the program: `MISMATCH in valu_int`,
`device_wall_s=0.000427`, `FAIL` (with the check: exit 3, `hipGetLastError() -> out of memory`). The first attempt's
stderr was not kept, so that this was that failure is the explanation that fits every fact, not a proof. Fixed in two ways: a module the host has no room
  for is `hipErrorOutOfMemory` (`load_failure`; test `test_amd_hip_module_oom`, run with the reserve set to more than
  any host has), and the benchmark program checks launch errors (it does since the first attempt above). A
  benchmark on a machine near its memory limit therefore fails loudly now, and the harness repeats the run.
* A counting run per workload, target and build gives exact instruction counts, one host thread, summed over the
  dispatches, from the shim's own probe: `VGPU_DISPATCH_STATS=<file>` (amd/src/hip_api.cpp) appends one line per
  finished launch with the `DispatchStats` numbers the profiler's SQ_* counters are made of, and needs no profiler
  (`--counts-via probe`, the default). `--counts-via rocprof` takes them from `rocprofv3 --pmc SQ_INSTS_* ...` inside
  `vgpu shell -y --gpu amd/<profile> --count 1 --no-isolate -c` instead, and `both` uses the probe's and requires
  rocprofv3's to equal them wherever rocprofv3 ran. Timed runs have no profiler.
* **Wave-instructions per CPU second** (Minst/s): counted instructions over the CPU time (all
  threads) of the device phase, which the program times around its launches and
  `hipDeviceSynchronize`. Launch and API workloads report calls per second; `startup` is the
  whole-process wall time. 9 runs after a warm-up, alternating over the configurations: one
  host thread (`VGPU_THREADS=1`) and the default (all CPUs the process may use: 8 here).
  With several threads CPU time is a sum, so the last column of the baseline gives the rate by
  wall clock, which is what a user sees.

*What counts.* The interpreter's total (`DispatchStats::instructions`) is not a profiler counter,
so the harness sums `SQ_INSTS_{SALU,SMEM,VALU,VMEM,LDS,BRANCH,SENDMSG}`, which omits SOPP
instructions that are none of those (`s_waitcnt`, `s_nop`, `s_endpgm`, `s_barrier`). Against
`amd_exec_bench`'s exact count (same quick-size launches) the sum is low by 0.09% on `valu_int`
(823040 vs 823808), 0.13% on `valu_f32` (568832 vs 569600), 0.02% on `divergent` (952528 vs
952720) and 11% on `wave` (311168 vs 350080: each `ds_bpermute` is followed by an `s_waitcnt`).

### Workloads

| workload | group | what it exercises |
| --- | --- | --- |
| valu_int | alu | 32-bit integer chain: multiply-add, xor, shifts, rotate, popcount |
| valu_f32 | alu | f32 fma chains plus unfused mul/add |
| valu_f64 | alu | f64 fma chains |
| mfma | matrix | `v_mfma_f32_16x16x16_f16` in a loop (MI300X only; RDNA2 has no matrix unit) |
| mem | memory | global memory: 16-byte vector copy, scalar scale, indexed gather |
| lds | memory | LDS with barriers: tree reduction, 32x32 tile transpose |
| atomics | memory | global atomic add (one address, 256 bins), max, a CAS loop |
| divergent | control | data-dependent `while` (collatz) and a `switch` |
| wave | control | `__shfl*` (`ds_bpermute`), `__ballot`, `__any`, `__all` |
| launch | api | 100000 one-wave launches on two streams |
| api | api | `hipMalloc`/`Memcpy`/`Memset`/`Free`, events, a stream: 13 calls a round |
| startup | api | one launch of a trivial kernel; read whole-process wall time |

The RX 6800 binary is wave32 (`gfx1030`), so its counts are about twice the MI300X's for the
same source. WMMA (RDNA3+) is not covered. Mix of what they run (full-size counts; MFMA is part
of VALU; SMEM is under 0.1% everywhere):

| workload | target | wave-instr | waves | VALU | SALU | VMEM | LDS | BRANCH | MFMA (in VALU) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| valu_int | mi300x | 8195840 | 256 | 81% | 13% | 0% | 0% | 6% | 0% |
| valu_int | rx6800 | 14343680 | 512 | 79% | 14% | 0% | 0% | 7% | 0% |
| valu_f32 | mi300x | 5637632 | 256 | 73% | 18% | 0% | 0% | 9% | 0% |
| valu_f32 | rx6800 | 12298752 | 512 | 75% | 17% | 0% | 0% | 8% | 0% |
| valu_f64 | mi300x | 8199424 | 256 | 62% | 25% | 0% | 0% | 12% | 0% |
| valu_f64 | rx6800 | 14350336 | 512 | 57% | 29% | 0% | 0% | 14% | 0% |
| mfma | mi300x | 2050368 | 64 | 25% | 50% | 0% | 0% | 25% | 25% |
| mem | mi300x | 4967168 | 768 | 54% | 18% | 18% | 0% | 9% | 0% |
| mem | rx6800 | 12042240 | 1536 | 62% | 15% | 15% | 0% | 8% | 0% |
| lds | mi300x | 3087680 | 512 | 14% | 33% | 15% | 21% | 17% | 0% |
| lds | rx6800 | 7265600 | 1024 | 11% | 30% | 30% | 16% | 14% | 0% |
| atomics | mi300x | 11955856 | 64 | 29% | 50% | 7% | 0% | 14% | 0% |
| atomics | rx6800 | 12391696 | 128 | 29% | 49% | 8% | 0% | 14% | 0% |
| divergent | mi300x | 17902446 | 1024 | 50% | 35% | 0% | 0% | 15% | 0% |
| divergent | rx6800 | 33425580 | 2048 | 46% | 35% | 0% | 0% | 19% | 0% |
| wave | mi300x | 6147968 | 128 | 50% | 33% | 0% | 12% | 4% | 0% |
| wave | rx6800 | 12295168 | 256 | 50% | 33% | 0% | 12% | 4% | 0% |

### Baseline

`tools/bench-amd/bench.sh --build build --cpus 24-31 --runs 9`, run twice in a row (A, B). A cell is
`median [best] spread` of Minst/s (1 thread); 8-thread columns are by wall clock.

| workload | target | wave-instr | 1 thread, run A: median [best] spread | run B | 8 threads by wall clock, A / B (median) | best 8-thread wall rate / best 1-thread rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| valu_int | mi300x | 8195840 | 3.10 [3.15] 3% | 2.02 [3.18] 71% | 23.7 / 14.2 | 7.6x |
| valu_int | rx6800 | 14343680 | 3.25 [3.31] 3% | 3.32 [3.33] 21% | 24.8 / 17.1 | 7.6x |
| valu_f32 | mi300x | 5637632 | 2.32 [2.35] 5% | 1.33 [2.11] 63% | 14.3 / 6.1 | 7.3x |
| valu_f32 | rx6800 | 12298752 | 2.30 [2.97] 46% | 2.92 [3.01] 44% | 15.6 / 15.1 | 7.5x |
| valu_f64 | mi300x | 8199424 | 2.82 [2.89] 26% | 2.80 [2.91] 52% | 21.5 / 12.9 | 7.5x |
| valu_f64 | rx6800 | 14350336 | 3.04 [3.08] 4% | 3.11 [3.16] 49% | 23.0 / 12.0 | 7.6x |
| mfma | mi300x | 2050368 | 0.91 [0.91] 2% | 0.83 [0.86] 25% | 7.0 / 3.3 | 7.7x |
| mem | mi300x | 4967168 | 2.49 [2.50] 8% | 2.36 [2.39] 9% | 19.1 / 9.2 | 7.7x |
| mem | rx6800 | 12042240 | 3.32 [3.37] 35% | 3.07 [3.17] 37% | 25.1 / 13.7 | 7.6x |
| lds | mi300x | 3087680 | 1.37 [1.77] 48% | 1.11 [1.65] 67% | 9.1 / 7.2 | 7.5x |
| lds | rx6800 | 7265600 | 2.97 [3.14] 38% | 3.08 [3.15] 7% | 19.0 / 14.6 | 7.5x |
| atomics | mi300x | 11955856 | 6.29 [6.36] 49% | 6.12 [6.25] 48% | 1.1 / 1.1 | 0.3x |
| atomics | rx6800 | 12391696 | 6.67 [6.71] 3% | 4.26 [6.49] 70% | 2.2 / 2.4 | 0.6x |
| divergent | mi300x | 17902446 | 5.15 [5.31] 9% | 3.50 [5.46] 66% | 28.5 / 15.5 | 7.3x |
| divergent | rx6800 | 33425580 | 3.75 [5.62] 62% | 5.56 [5.75] 45% | 14.9 / 23.4 | 7.4x |
| wave | mi300x | 6147968 | 1.88 [2.26] 27% | 3.03 [3.08] 38% | 8.4 / 16.1 | 7.5x |
| wave | rx6800 | 12295168 | 2.32 [2.84] 23% | 3.99 [4.03] 2% | 9.8 / 26.0 | 7.5x |
| launch | mi300x | | 304 [334] 28% klaunch/s | 458 [467] 5% klaunch/s | 304 / 463 klaunch/s (all threads) | |
| launch | rx6800 | | 239 [248] 26% klaunch/s | 370 [377] 4% klaunch/s | 237 / 373 klaunch/s (all threads) | |
| api | mi300x | | 432 [1004] 196% kcall/s | 1265 [1426] 20% kcall/s | 580 / 1245 kcall/s (all threads) | |
| api | rx6800 | | 1113 [1280] 23% kcall/s | 1228 [1423] 105% kcall/s | 1114 / 1020 kcall/s (all threads) | |
| startup | mi300x | | 0.002 s [0.002] | 0.002 s [0.002] | | |
| startup | rx6800 | | 0.005 s [0.005] | 0.005 s [0.005] | | |

Reading it. The two runs' bests agree within 3% on 9 of 17 single-thread rows and within 7% on
15; the exceptions are `wave` (run A never got a clean run: 1.9 vs 3.1 on the MI300X) and
`valu_f32` on the MI300X (11%). Medians moved by up to 1.7x between runs at load 30 to 50, so use
the larger best. In round numbers, the interpreter runs **about 3 Minst/s per host thread on
VALU code** (valu_int 3.3, valu_f64 2.9 to 3.2, valu_f32 2.1 to 3.0), i.e. about 300 ns per
64-lane wave-instruction; **`mfma` 0.9** (a quarter of its instructions, nearly all its time),
**`lds` 1.7** on the MI300X (3.1 on the RX 6800), `mem` 2.4 to 3.4, `wave` 3.1 to 4.0, and the
scalar- and branch-heavy `atomics` (6.3 to 6.7) and `divergent` (5.2 to 5.7) are fastest.
An RDNA wave32 instruction costs as much as a wave64 one (the lane loops run over 64 lanes
with the upper 32 masked off), so per lane it is twice as expensive. With 8 CPUs every workload
but `atomics` is 7.3x to 7.7x faster by wall clock (best over best) at 3 to 16% more CPU time.
`atomics` is the exception, slower with threads (below). `launch` is 300 to 460 klaunch/s,
`api` 1.0 to 1.4 Mcall/s, `startup` 2 ms (MI300X) and 5 ms (RX 6800), which is the whole process.

### The host refusing a thread

`execute()` spreads a grid over host threads and a stream's work runs on a host thread of its own. Where the host refuses to
start one (EAGAIN at a process or user limit, or no address space left; reproduced with `ulimit -v` and `RLIMIT_NPROC`), the
pool's vector was destroyed with a joinable `std::thread` in it, which is `std::terminate` (the program printed "terminate
called without an active exception" mid-launch), and a stream's first `submit` threw `std::system_error` out of
`hipLaunchKernel`, an extern "C" entry point. Now a range of work-groups whose thread is refused runs on the launching thread
(the ranges are independent, so the answers and counts are the same; `start_or_run_here`), the lockstep pool, whose workers
wait on one another, fails the launch with an `Err::Internal` that names the cause, and a refused stream thread is
`hipErrorOutOfMemory` with the stream left usable. Tests: `test_amd_gcn_thread_refused` (an 8-group dispatch at
`VGPU_THREADS=4` under `RLIMIT_NPROC=1` still gives the right answers and counts all 8 waves) and
`test_amd_hip_thread_refused`; both say SKIP and pass where the process is exempt from the limit (root). Mutation checks: with
the fallback removed the first dies with the `std::system_error`, with the catch removed the second aborts, with the DS guard
removed `test_amd_gcn_same_bits` segfaults.

### The counter check

`--check-counters` runs the workloads at the `quick` size on one host thread, counted by the shim's
`VGPU_DISPATCH_STATS` probe (and, where rocprofv3 is there and the shim is not sanitizer-built, by rocprofv3 too, which
must agree: `--counts-via both`, what the ctest does), and compares the `SQ_INSTS_*`, `SQ_WAVES`, `SQ_INSTS_VMEM_RD/WR` and `TA_FLAT_ATOMIC_WAVEFRONTS` totals,
per workload and target, with `tools/bench-amd/expected/amd-hip-7.2.26015.json` (counts depend on
the compiler, so one file per hipcc release). It is the ctest test `bench_amd_counters` (about 6 s),
and skips (`SKIP:`) without a ROCm that has `hipcc` or without an expected file for the release. A sanitizer-built
shim, or no rocprofv3, no longer skips: the probe counts alone are compared (a NOTE says the rocprofv3 cross-check
was not made). The counts at one
thread are deterministic, including the `atomics` CAS loop: `--update-expected` followed by two
`--check-counters` matched all 21 workload/target pairs. Mutation check: raising one stored count
by one makes it exit 1 with `mi300x/valu_int: SQ_INSTS_VALU expected 666625, got 666624`.

*Why rocprofv3 "wrote no counter CSV" on server2.* `vgpu shell` puts its own `rocprofv3` first on PATH (a wrapper that
runs ROCm's tool with `--rocm-root` pointing at VirtualGPU's `librocprofiler-sdk.so.1`), and writes that wrapper only
when `shim/librocprofiler-sdk.so.1` exists, i.e. when the `vgpurocprof` target was built. A build of only `vgpu` and
`vgpuhip` (what a quick benchmark build is) has no wrapper, so the bare `rocprofv3` of the counting run is ROCm's own: it
loads ROCm's profiler SDK over the simulated agents, finds no counters, writes nothing at all, and the program still
passes (rc 0, `PASS`), which is why the harness could say only "no counter CSV". Reproduced on server2 by removing
the library from a copy of the shim (no file written), and by running `/opt/rocm/bin/rocprofv3` by path inside the
shell (the same). With `vgpurocprof` built, rocprofv3 wrote its CSV in all of about 2,500 counting runs under load
(4 at a time, 1 and default threads), and the harness counted all 21 workload/target pairs identically to
`expected/` by both routes (`counters match ... counted by both, rocprofv3 agreed on 21`). The harness now says which
library is missing, and its default counts need no profiler, so "counters identical" is a real check on any host.

### Profiles

`perf` is unusable on server2 (`perf_event_paranoid` 4, no sudo: "Failure to open any events").
The profiles come from `tools/bench-amd/profile/sigprof.c`, an `LD_PRELOAD` sampler (an `ITIMER_PROF`
timer, 2000 Hz here, recording the interrupted pc and with `SIGPROF_STACK=1` ten caller frames) and
`report.py` (`addr2line`; `--within`, `--inclusive`, `--lines`, `--callers-of`). The simulator was
built with `-g` and the same `-O3`/LTO. All are `VGPU_THREADS=1`, full size, 1700 to 5500 samples
inside `Machine::step` each (that filter leaves out the program's own host reference). A sample is
charged to the function containing the instruction, so inlined callees count toward their caller.
libc has no symbols here: the entry labelled `__nss_database_lookup` is `memcmp` (its callers by
the recorded chains are `OpName::rfind`, `std::string::rfind`/`compare`/`find`).

Share of the samples inside `Machine::step`, MI300X:

| workload | samples | name machinery | operand reads | libc + libgcc | `Inst` ctor/dtor |
| --- | ---: | ---: | ---: | ---: | ---: |
| valu_int | 2543 | 26.7% | 28.2% | 7.8% | 1.9% |
| valu_f32 | 2460 | 22.3% | 29.4% | 4.1% | 1.3% |
| valu_f64 | 2750 | 32.9% | 3.3% | 3.9% | 1.2% |
| mfma | 2209 | 4.2% | 0.0% | 40.3% (`__extendhfdf2`) | 0.4% |
| mem | 1945 | 16.1% | 21.6% | 4.0% | 1.1% |
| lds | 1753 | 47.6% | 3.8% | 7.5% | 0.7% |
| atomics | 1850 | 33.0% | 9.9% | 5.0% | 2.0% |
| divergent | 3114 | 41.0% | 11.2% | 5.0% | 2.2% |
| wave | 1872 | 26.9% | 23.9% | 7.8% | 1.2% |

*Name machinery* is the `OpName` class, `gcn_exec.cpp` lines 90 to 125: `name_hash` (line 100, an FNV
hash of the instruction's name taken on every `OpName` construction) and `operator==` (line 111, once per
rung of every `if (op == "..."_op) ... else if` ladder). *Operand reads* are `scalar` (559 to 603) and
`lane_src` (638 to 645): a source that is not a VGPR goes through the `scalar` switch once per lane
though it is the same in every lane. (Line ranges are read from the profile's line view.) On the RX 6800 the name machinery is
34% (`valu_int`), 29% (`valu_f32`), 40% (`valu_f64`), 47% (`lds`, `divergent`), 43% (`atomics`).

Top functions per workload (self time inside `step`; "lane loops" are the `each` lambdas of `vector_alu`):

| workload | top functions |
| --- | --- |
| valu_int | lane loops 23.2%, `OpName::operator==` 22.3%, `vector_alu` 11.8%, `scalar` 11.3%, memcmp 6.3%, `step` 5.3%, `more_alu` 5.0% |
| valu_f32 | `OpName::operator==` 19.0%, `vector_alu` 15.3%, `lane_src` 13.1%, `scalar` 13.0%, lane loops 10.6%, `step` 4.9% |
| valu_f64 | lane loops 33.9%, `OpName::operator==` 27.0%, `step` 9.1%, `vector_alu` 6.2%, `scalar` 4.5%, `sgpr` 4.4% |
| mfma | `matrix_multiply` 52.0%, `__extendhfdf2` 37.8%, `step` 3.0% |
| mem | lane loops 20.3%, `scalar` 13.1%, `OpName::operator==` 11.6%, `MemoryManager::scalar_location` 7.5%, `step` 5.9%, `sgpr` 5.9%, `global_access` 4.3% |
| lds | `OpName::operator==` 41.8%, `lds_access` 14.6%, memcmp 7.5%, `step` 6.9%, `scalar_location` 5.8% |
| atomics | `step` 18.1%, `OpName::operator==` 17.2%, `scalar_location` 7.1%, `scalar_alu` 5.6%, `lane_src` 5.0% |
| divergent | `OpName::operator==` 31.0%, `step` 11.5%, lane loops 11.4%, `vector_alu` 8.7%, `scalar` 5.4% |
| wave | `OpName::operator==` 23.5%, lane loops 12.9%, `lds_access` 10.6%, `scalar` 9.1%, `lane_src` 5.6%, `compare` 4.8% |

**`launch` and `api`** are not interpreter-bound: `Machine::step` is 39% inclusive of the `launch`
samples; the rest is the work-queue hand-off (`WorkQueue::submit`, condition variables), `set_up_group`
(12.6%, the register-file set-up of each one-wave launch) and `MemoryManager::free`. `api` is about 90% hand-off and
allocator.

**`atomics` with threads.** `memory_atomic_lock` (`gcn_exec.cpp` line 8244, 251 striped mutexes by address) is
taken for every lane of every atomic; a workload that hits one address from every work-group queues the
host threads on one mutex. At 8 threads 87% of the samples (quick size, 315 samples) are in `pthread_mutex_lock`,
`__lll_lock_wait_private` and `__lll_lock_wake_private`, the run takes 10.9 to 11.1 s of wall clock against
1.9 s on one thread (5.6x to 5.9x slower), at 65 to 75 s of CPU. Counts also stop being deterministic there, which is
why the counter check is at one thread.

### What is re-derived on every execution and could be decoded once

`DecodeCache` decodes each instruction once per module and `fetch` is cheap (1 to 3%). What the decoded
`gcn::Inst` lacks is the operation's identity in a form the executor can use:

1. **The operation as a name.** Every helper builds an `OpName` (FNV over the characters) and compares against
   precomputed literal hashes rung by rung. An instruction `vector_alu`'s own ladder handles has first failed
   the 118 rungs of `more_alu` (called at line 3498), and `rdna_alu`/`cdna5_alu` before that on those parts;
   `vector_alu` alone has 217 rungs, the file 1050. `rdna_alu` and `cdna5_alu` also copy the name into a
   `std::string`, strip the suffix and hash it again, per instruction (lines 1935 to 1938, 2384 to 2387).
2. **Classifications from the name's text:** `rfind("v_mfma", 0)`, `rfind("v_cmp", 0)`, `find("f64")`, `find("_u16")`,
   `is_plain_store`'s `find("_store")`, `as_short` assembled as a string in `vector_alu`. Each is a `memcmp` or scan per execution.
3. **Operand classes.** Whether a source is a VGPR or lane-invariant is known at decode, but `lane_src`
   re-tests it and calls `scalar` per lane.
4. **Per-instruction frame:** a default-constructed `Inst remapped`, `Narrowed`, `FloatMode`, `debug::active()`, `tracing()`.
5. **Matrix operands:** `matrix_multiply` allocates three `std::vector<double>` per MFMA, decodes every element
   of A, B and C through a chain of `type ==` tests, and does 4096 double multiply-adds for a 16x16x16.

### Tried on scratch copies (not committed)

Each is a patched copy of the tree built in its own directory; `bench.sh --build base --build patched --runs 9 --threads 1
--cpus 24-31`, MI300X, counters identical in every case (the harness prints `COUNTER DRIFT` otherwise), results
`PASS`. Ratio = patched / base Minst/s (>1 faster); p is Mann-Whitney.

| experiment | workload | median ratio | best ratio | p | load | verdict |
| --- | --- | ---: | ---: | ---: | --- | --- |
| X1+X4: half to double by bit manipulation in `matrix_multiply`'s `value` (exhaustive check: all 65536 patterns equal the compiler's conversion, NaNs and subnormals included), and `Inst remapped` as `std::optional<Inst>` | mfma | 1.024 | 1.093 | 0.136 | 10-22 | same |
| | valu_int | 1.009 | 0.969 | 0.796 | | same |
| | valu_f32 | 0.944 | 0.923 | 0.006 | | slower |
| | valu_f64 | 0.949 | 0.943 | 0.546 | | same |
| | mem / lds | 1.006 / 0.989 | 1.106 / 1.000 | 0.80 / 0.34 | | same |
| | atomics | 0.950 | 0.948 | 0.000 | | slower |
| | divergent | 0.943 | 0.952 | 0.000 | | slower |
| | wave | 0.965 | 0.973 | 0.001 | | slower |
| X5: the name hash cached in the `Inst` (every `OpName(in.name)` site and `vector_alu`'s; a throwaway version keyed by the string's address, single-thread only) | valu_int | 1.071 | 1.012 | 0.546 | 10-13 | same |
| | valu_f32 / valu_f64 | 1.008 / 0.956 | 1.012 / 0.957 | 0.04 / 0.000 | | same / slower |
| | mem, lds, atomics, divergent, wave | 0.99 to 1.01 | 0.999 to 1.013 | 0.02 to 0.19 | | same |
| X6: `lane_src` switches on the operand kind and reads VGPRs, inline constants and literals directly (no `scalar` call) | valu_int | 1.083 | 1.108 | 0.161 | 27-46 | same |
| | valu_f32 | 0.873 | 0.960 | 0.436 | | same |
| | mem | 1.062 | 1.075 | 0.024 | | faster |
| | wave | 1.049 | 1.088 | 0.387 | | same |
| | divergent | 1.047 | 1.046 | 0.000 | | faster |

What this says, and where the profile misled:

* **X1**: `__extendhfdf2` vanished from the profile (it was 37.8% of `mfma`), yet the speed-up is not significant
  (median 2%, best 9%). A profile of the patched build puts 555 of 2074 samples in the `value` lambda itself: the
  libgcc call was a small part of the per-element decode, so the cost is `value()`'s type chain and the vectors, not the conversion. My
  first estimate (1.5x to 1.7x) was wrong. The libgcc routine's hot instruction is the one after `stmxcsr`, so
  skid made it look like the whole cost.
* **X4** (lazy `Inst remapped`) is slower, 5 to 6% on four workloads at p <= 0.006 and faster on none: dropped. The profile's
  1 to 3% for `Inst` construction was not a saving.
* **X5** is no change (within 1.3%, one 4% slowdown on `valu_f64`) even though `name_hash` was 5 to 16% of the samples on the
  workloads it should help. The compiler may already be hoisting or the sampled lines were skid from the ladder. Dropped; an earlier
  version of X5 that missed `vector_alu`'s construction site was also within noise.
* **X6** is a small gain on `mem` and `divergent` (about 5%, p <= 0.024) at load 27 to 46, noise elsewhere: promising but
  needs a quiet-machine confirmation before it counts.

The lesson for whoever does item 1 to 3 below: shares of samples overstate what a change removes, and the
name ladder (line 111) is the only item whose share is large and has not been tried.

### Ranked cheap wins

Estimates are what a sample share could at most give, discounted by the lesson above; none of these was applied
except where the table above says so. Every change needs the same-bits test and the A/B rule.

| # | change | evidence | est. gain | risk |
| --- | --- | --- | --- | --- |
| 1 | Replace the `op == "..."_op` ladders by a `switch` on the hash (`consteval` case labels, a compile-time check that no two rungs collide) or an opcode id assigned at decode; or at minimum order `more_alu`'s 118 rungs behind a cheap class test | line 111 is 12 to 48% of samples (median 26%; `lds` 42%, `divergent` 31%, RX 6800 `valu_int` 28%); not tried, since it touches about a thousand rungs | 10 to 30% on ALU-bound kernels, more on `lds` (untested; X5 shows the hash itself is not the cost, the compares are) | medium: mechanical but large; `||` rungs and fall-through to later helpers; counter check plus a test executing every opcode both ways |
| 2 | Decode-time flags (`is_mfma`, `is_cmp`, `is_f64`, `is_store`, short-form name) instead of the `rfind`/`find` text tests; no name copy in `rdna_alu`/`cdna5_alu` | `memcmp` 4 to 10% of samples; RDNA copies the name per instruction | 3 to 8%, more on RDNA | low |
| 3 | Host atomics (`__atomic_*`) for aligned 32/64-bit add/min/max/and/or/xor/exchange/cas instead of the striped mutex | 87% of samples at 8 threads in the mutex; 5.6x to 5.9x slower in wall clock than one thread | `atomics` at the default thread count: 5x or more; none at one thread | medium: ordering, float and packed atomics; counts only deterministic at one thread |
| 4 | Lane-invariant operands read once per instruction (SGPR, constant, literal): finish X6 into the `each` helpers | 22 to 29% operand-read share on `valu_int`, `valu_f32`, `mem`, `wave`; X6 alone measured about 5% on two workloads | 5 to 15% on VALU-bound kernels | medium: DPP, SDWA `sel`, `abs`/`neg`, `narrow_dst` all pass through `lane_src` |
| 5 | `matrix_multiply`: no per-MFMA heap vectors, element decode hoisted by operand type, bit-conversion of halves (X1) | `mfma` is 25% of its instructions and ~90% of its time; X1 alone gave 2 to 9% | up to 1.5x on `mfma` (untested beyond X1) | low to medium: exhaustive conversion test; results compared bit for bit on rocWMMA/rocBLAS paths |
| 6 | Contiguous-lane fast path for global memory (one allocation lookup per wave) | `scalar_location` + `load_scalar` 10% of `mem`, 6 to 8% of `lds`/`atomics` | 5 to 12% on memory workloads | medium: the memory layer is shared with the CUDA engines |
| 7 | Cheaper one-wave launch (`set_up_group`, `VgprFile` zeroing) | 12.6% of `launch` samples | 5 to 10% of launch rate | low |
| 8 | `popcnt` target attribute for `__popcountdi2` (1.4% of `valu_int`) | small | 1 to 2% | very low |

### Ranked wins 1, 2, 4 and 8 applied (`speedup2-amdops`)

Branch `speedup2-amdops` from `speedup` a12f8d84. Same machine, harness and pinning as above (server2, CPUs
26-31, `VGPU_THREADS=1`, 11 alternating runs, Mann-Whitney p; load average 4.7 to 26 per workload, median 8);
base is a12f8d84 built Release in its own directory, B the branch. Counters identical on every row
(`--check-counters`: 21 of 21 pairs match).

| workload | target | median ratio | best ratio | p |
| --- | --- | ---: | ---: | ---: |
| valu_int | mi300x / rx6800 | 2.11 / 2.59 | 2.16 / 2.58 | 0.000 |
| valu_f32 | mi300x / rx6800 | 2.11 / 2.82 | 2.04 / 2.97 | 0.000 |
| valu_f64 | mi300x / rx6800 | 1.89 / 2.66 | 1.93 / 2.67 | 0.000 |
| mem | mi300x / rx6800 | 1.51 / 1.90 | 1.51 / 1.92 | 0.000 |
| lds | mi300x / rx6800 | 2.24 / 2.33 | 2.28 / 2.33 | 0.000 |
| atomics | mi300x / rx6800 | 1.86 / 2.34 | 1.84 / 2.35 | 0.000 |
| divergent | mi300x / rx6800 | 2.51 / 3.03 | 2.50 / 3.12 | 0.000 |
| wave | mi300x / rx6800 | 1.68 / 1.81 | 1.68 / 1.94 | 0.000 |

What changed:

* **Win 1, the ladders.** `amd/tools/ladder-to-switch.py` rewrote every `if (op == "a"_op || ...) ... else if` chain of
  `gcn_exec.cpp` whose rungs test only names (783 case labels, 502 rungs, 30 chains; a chain with an impure rung in the
  middle keeps that rung as an `if` and converts the pure runs either side) into `switch (op.hash())`. The case keeps the
  rung's original condition, tested verbatim, so a hash collision still falls to the else-branch as the text comparison did;
  `if` statements in a row count as one chain only where each body returns. The SOPP tail of `Machine::step` (about 30
  rungs, each constructing and hashing an `OpName`) is a hand-written switch in the ladder's order. A first A/B of this alone
  (`valu_int` 1.55x, `lds` 1.93x, `divergent` 2.17x, `mem` 1.20x, p <= 0.003) is what the sample shares of the profile above
  said could be had; the rest of the table adds wins 2 and 4.
* **Win 2, decode-time facts.** `gcn::finalize` (called at the end of `gcn::decode`, halves of a VOPD included) fills in
  `Inst::name_hash`, the short form's name and hash that `vector_alu` built as a `std::string` on every instruction, and
  `InstFlag` bits for the `rfind`/`find` tests of `step` (`v_mfma`, `v_cmp`, `f64`, `_store`, `_atomic`, `ds_gws_`, ...). An
  `Inst` not decoded this way (`flags == 0`) works them out as before. `rdna_alu` and `cdna5_alu` take a view of the name
  instead of copying it.
* **Win 4, lane-invariant sources.** `InvariantSources` reads an SGPR, inline constant or literal source once per
  instruction (in `vector_alu` and `compare`) and `lane_src` returns it from a four-entry table; the global-access loop reads
  its scalar base once. Alone (A = the branch with the table switched off, 11 runs, MI300X): `valu_int` 1.37 (p 0.000),
  `valu_f32` 1.12 (p 0.001), `mem` 1.21 (p 0.010), `wave` 1.09 and `divergent` 0.96 (not significant).
* **Win 8, POPCNT: dropped.** A `target("popcnt")` copy of the `v_bcnt_u32_b32` lane loop: median ratio 1.013 on `valu_int`
  (p 0.61), 1.004 to 1.018 on the others (p 0.056 to 0.70). Under the 2% bar and not significant; removed.

The same-bits test is `test_amd_gcn_same_bits` (`amd/tests/unit/test_gcn_same_bits.cpp`, ctest-wired as a unit test): every
instruction shape of the seven ISA corpora and of a SOPP corpus (15,566 shapes, `amd::digest_instruction` in
`gcn_exec.cpp`) runs once on a wave of pseudo-random registers, LDS, scratch and memory in four ways (any bits, small
integers, floats and doubles with the special values among them, half the lanes off; branches are sent to a distant target)
and a digest of all registers, masks, LDS, memory, counts and error text is compared with
`amd/tests/data/same_bits_digests.txt`, which was recorded from the unmodified a12f8d84 plus only the probe. Mutation
checks (each makes it fail): two case labels swapped (123 shapes differ), the invariant table off by one (3,088), a wrong
decode flag (1,527), `s_cbranch_scc0` inverted (6). Instructions that read the clock are recorded as unstable and skipped;
`ds_nop`, `ds_consume` and `ds_append` are skipped because the interpreter crashed on them with the fuzz corpus's operands:
the decoder gives these three no address operand (`Inst::src` is empty), and `lds_access` read `in.src[0]` off the end of an
empty vector (a SEGV in `lane_src`, found with a signal handler on the base commit f81cc0c1: all three shapes of the
gfx1250 corpus die there). They stay out of the digest, because the base crashes and so no base digest of them can exist;
`ds_nop` now does nothing (it counts as an LDS instruction) and `ds_append`/`ds_consume` throw `Err::Unsupported` ("has
no address operand and is not modelled"; the wave-wide counter at M0 is not documented well enough to model), checked by
`ds_instructions_without_an_address_fail_gracefully` in the same test file on RDNA2, RDNA3, RDNA4 and gfx1250.

Two things the digest caught that a pass over the ladders would not have: GCC expands `floor`, `ceil` and `trunc` in line
where it thinks the code hot (a signaling NaN then comes back unchanged; libm returns a quiet one), and which places those
are moved with the switch. The vector instructions (`v_floor/ceil/trunc_f32/f64`) were in-line before and are now
`floor_inline` and friends with that behaviour spelled out; the scalar ones were libm and `-fno-fp-int-builtin-inexact`
keeps them so. Likewise which of two NaN operands an `fma` or a double add propagates follows the order the compiler gives
its operands: `v_fma_mix*_bf16` and `v_dot2c_f32_f16` needed their operands written in an order (and, for the dot product,
held there with an empty `asm`) to give the bits the old build gave. Other sites with two NaN operands of different
payloads (`fmin`/`fmax`, the other `fma`s) may be compiled differently now; the corpus run with these inputs found none
beyond those two, but it cannot prove there are no more.

Tests run on server2 against the branch: `test_amd_*` (all 38 unit binaries that run there: pass), ctest `amd|gcn|rocprof|
hipcc|hip_` (all pass or skip but `amd_rocprofiler_abi` (rocprofiler-sdk header version), `amd_rocprofv3` and `amd_tools`'
user-namespace steps, which fail for the host, not the change), `bench_amd_counters` (21 of 21), PyTorch tests skip there.

### Not done, and not measured here

* The `std::optional<Inst>` change (X4) and the cached hash (X5) are negative results and stay out; X1 and X6 are inconclusive or small.
* Loops over 64 lanes with a mask test (wave32 pays for 32 masked lanes), `ctz` lane iteration: not measured.
* RDNA3/4 `v_wmma_*`, other MFMA shapes, gfx1250: not timed here; see "The AMD matrix instructions, every form" (branch `speedup4-amdmatrix`).
* The effect of any ranked change on pantheon, PyTorch or rocBLAS workloads; those spend more time in
  `matrix_multiply`, the memory manager and `lds_access` than these microbenchmarks.
* Host hardware counters (no PMU access here).


## Using this to check a change

1. Build baseline and candidate in separate build directories (same flags).
2. `tools/bench/bench.sh --check-counters --build <candidate>` (or ctest
   `bench_counters`): counts must not move.
3. `tools/bench/bench.sh --ab <baseline> <candidate> --runs 9 --workloads <group>`
   with nothing else of yours running; note `uptime`. Quote median, best and p,
   not one number.
4. Add a same-bits test for what you touched (tests/unit/test_fastpath.cpp is the
   pattern) and show it fails when the change is wrong.

## Not measured

This list is the baseline's; later rounds measured some of it, as noted.

* Host instruction counts and IPC (no PMU on this VM; unknown on hosted runners).
* `VGPU_THREADS` > 1 scaling in the baseline (harness supports `--threads`); later host-thread measurements
  are in [perf-design/host-parallelism.md](../host-parallelism.md).
* Other GPU profiles / architectures in the baseline (the program is compiled for sm_86); the SASS engine
  now runs every generation from sm_75 to sm_120a, and [perf-design/round10-real-workloads.md](../round10-real-workloads.md)
  runs programs on their own generation's profile.
  The AMD engine has its own section, "The AMD engine".
* The pantheon soak workloads outside the optional mode above, and the PyTorch/cuBLAS paths in the baseline.

