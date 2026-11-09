## The AMD runtime: calls, queues and work-group threads (branch `speedup4-amdmt`)

What a HIP call, a stream's hand-off and a launch's threads cost, measured and cut where it could be cut without moving a result.
Base is `speedup` f81cc0c1 (Tier 1 and Tier 2). Measured on server2 (Ryzen 9 9950X, 32 logical CPUs shared with other people's builds
and benchmarks), Release, GCC 13.3, `tools/bench-amd` pinned to CPUs 24-31, simulated MI300X and RX 6800 only (no card touched), A and
B alternating, 7 to 21 runs. The final A/B below ran at a 1-minute load of 5 to 9 (median 6); the leave-one-out runs at 7 to 15. Every
ratio is the median wall time of A over B (above 1: B is faster), p the harness's Mann-Whitney p. Counts and bits cannot move in these
changes, and `test_amd_gcn_same_bits` (the guard, 15,566 shapes against digests recorded from the unmodified base) passes unchanged.

### New rows in the harness (`tools/bench-amd`)

All self-checking, none counted by the profiler (`--no-counts`; rocprofv3 writes no counter CSV on this host).

| workload | what it does | units |
| --- | --- | --- |
| `apicheap` | `hipGetDevice`, `hipGetLastError`, `hipPeekAtLastError`, `hipDeviceGetAttribute`: the fixed cost of a HIP call | calls |
| `launch_mid` / `launch_wide` | launches of 128 / 1024 one-wave work-groups of a tiny kernel on one stream: the hand-off to the work-group threads | launches |
| `launch_sync` | four chained one-wave launches, then `hipStreamSynchronize`: the hand-off between a stream's thread and its caller | calls |
| `memcpy` | 16 KiB and 64 KiB `hipMemcpy` (H2D, D2D, D2H), `hipMemcpyAsync`, `hipMemsetAsync`, `hipMemset`, each round ending in a stream synchronize | calls |
| `stream` | `hipStreamCreateWithFlags`, `hipEventRecord`, `hipStreamWaitEvent`, two launches, queries, synchronize, `hipStreamDestroy` | calls |
| `graph` | a graph of four chained kernel nodes captured from a stream, launched in a loop | launches |

### Where a launch's time goes

`launch` (100,000 one-wave launches on two streams) at one work-group thread, `tools/bench-amd/profile` on a `-g` build of this
branch (4000 Hz, call chains, so inclusive): the two stream threads are 94% of the samples and the calling thread 12%, so the rate is
the stream thread's per-launch work and not the hand-off. Inside it: `execute` 49% (the work-group's set-up, `set_up_group`, 14%, the
one wave's instructions 28%), `MemoryManager::free` 20% (the kernarg segment's, with its quarantine record), `find_ordinary_miss` 13%
(every allocation and free changes the table lock's epoch, so each kernel's first accesses look the table up again); half of the
samples have no source line, which is libc (malloc and free, the zeroing of the 64 KiB register file and of the kernarg chunk, futex). `strace -c` on 2,000 launches:
2.7 futex calls a launch on the base, and 1.0 `gettid` and `getpid` each, which are the main-thread check of `launch_in_order`. An
experiment that is not shippable (a scratch kernarg segment allocated once per thread and never freed, which moves every later device
address and the quarantine) makes `launch` 1.65x (MI300X) and 1.42x (RX 6800) faster, p < 0.001: the allocate-write-free of the
kernarg segment is about a third of a launch. Switching telemetry off (`VGPU_TELEMETRY=0`, 15 alternating pairs) is worth 2.8%.
`alloc`+`free` of 128 bytes alone is 147 ns on one thread; with the 192-byte write into the fresh allocation, which makes a 64 KiB
chunk real and frees it again, 455 ns.

### What changed

* **A HIP call with nothing watching is one load** (`ApiCall`, `hip_api.cpp`). The previous round's argument capture was already built
  only after the profiler test, but each call still read a function-static, a thread-local depth and the profiler pointer, and the
  profiler side was inline at ~560 call sites. One relaxed load of `g_api_watch` (0: nothing; 1: a profiler attached or `VGPU_TRACE_API`;
  2: not settled yet) now decides, and entering and leaving are out of line, cold. A profiler attaching sets 1, so calls made after that
  count their depth and report as before; one attaching while another thread is inside a call sees that call's nested calls reported as
  outermost, which is the one difference.
* **Hand-offs spin before they sleep** (`hip_queue.hpp`). A thread that waits for the other side of a stream's queue (the caller for a
  result, the stream's thread for work) polls for 400 pauses (a few microseconds) before it takes the futex, with `VGPU_QUEUE_SPIN=0` for none
  (it is also none when the process may use one CPU). The queue has a condition variable for each side, a notify that is a no-op with no
  sleeper, and a thread that is spinning stops at once when its queue is destroyed (without that, `hipStreamDestroy` waited out the spin,
  and `stream` was 14% slower).
* **A waited-for call runs on the calling thread when its stream is idle** (`WorkQueue::run_inline`, used by `in_order`). `hipMemcpy`
  and friends wake the stream's thread, sleep, and are woken in turn; with every item done, nothing running and every marker finished
  the work is numbered, finished and counted as a queued item is, on the caller. Anything queued meanwhile waits for it, a failure is
  the queue's as before, a profiler being attached (its copy records carry the thread) keeps the stream's thread. A stream used only
  this way never starts its thread. Not done for asynchronous items: they must return before the work does.
* **A launch's work-group threads are kept** (`gcn_pool.hpp`, used by the three places `execute` started and joined `std::thread`s).
  The same static ranges, a thread for each task and all of them alive at once (work-groups may wait on one another), the first task
  on the calling thread, the floating-point environment and signal mask of the launching thread, made afresh in a forked child. The
  CPUs a program may use stay the limit (`worker_count`).
* **Smaller things.** A wave's work-item ids are counted up, not divided out per lane (three 64-bit divisions a lane before); the
  main-thread check in `launch_in_order` is asked once per thread and fork, not with two system calls a launch; a device-to-device
  copy's buffer is no longer zero-filled first.

### Results

Base against this branch, 9 runs, one host thread and all of them (CPUs 24-31), load 5 to 9. p < 0.001 unless given.

| workload | MI300X, 1 thread | MI300X, 8 | RX 6800, 1 | RX 6800, 8 |
| --- | ---: | ---: | ---: | ---: |
| `api` (13 calls, 5 of them hand-offs) | **3.29x** | **3.32x** | **3.27x** | **3.17x** |
| `memcpy` | **2.22x** | **2.20x** | **2.21x** | **2.13x** |
| `launch_sync` | **1.23x** | **1.24x** | 1.18x (p 0.004) | 1.16x (p 0.003) |
| `stream` | 1.16x | 1.14x | 1.19x | 1.19x |
| `apicheap` (about 30 ns a call) | 1.08x | 1.10x | 1.09x | 1.10x |
| `graph` | 1.04x | 1.04x | 1.04x | 1.04x |
| `launch_mid` (128 work-groups) | 1.04x (p 0.03) | **2.51x** | 1.02x (p 0.01) | **1.88x** |
| `launch_wide` (1024 work-groups) | 1.07x | 1.11x (p 0.49) | 1.03x (p 0.003) | 1.19x (p 0.008) |
| `launch` (one wave, two streams) | 1.01x (p 0.55) | 1.02x (p 0.09) | 1.04x (p 0.30) | 1.05x (p 0.34) |

`launch` is the rate of the stream thread's own work (above), and none of this touches it. CPU time goes down where the hand-offs were
the cost (`api` 1.23 to 0.96 s, `launch_mid` with all threads 2.40 to 1.91 s) and up where a thread now polls before it sleeps:
`stream` +30%, `memcpy` +14%, `launch_sync` +10%, in processes that last 0.1 to 0.4 s; `VGPU_QUEUE_SPIN=0` gives that back and loses
the wall-clock gain below.

The same tree against itself with one change removed (11 to 21 runs, load 7 to 15; `ApiCall`'s single load touches only `apicheap`, whose
base-against-branch row above is its effect, 1.08x to 1.10x):

| change | workload | MI300X | RX 6800 |
| --- | --- | ---: | ---: |
| spinning (`VGPU_QUEUE_SPIN` 400 against 0) | `api` | 2.49x | 2.51x |
| | `memcpy` | 1.82x | 1.77x |
| | `launch_sync` | 1.18x | 1.13x |
| the queue's changes together (spinning, waited-for calls on the caller, the separate counters) | `api` | 3.35x | 3.23x |
| | `memcpy` | 2.24x | 2.24x |
| | `stream` | 1.14x | 1.14x |
| | `launch_sync` | 1.10x (p 0.24) | 1.15x |
| the pool (against a thread for each task, one launch) | `launch_mid`, 8 threads | 2.40x | 1.81x |
| | `launch_wide`, 8 threads | 1.05x (p 0.01) | 1.06x (p 0.02) |
| ids counted up + the main-thread check cached | `launch_wide`, 1 thread | 1.10x (p 0.003) | 1.04x |
| | `launch` | 0.99x | 1.07x (p 0.04) |
| the device-to-device buffer not zeroed | `memcpy` | 1.04x | 1.05x |

The workloads this does not touch (`valu_int`, `valu_f32`, `mem`, `lds`, `atomics`, `divergent`, `wave`, `mfma`, 7 runs, one thread
and all, both cards) moved between 0.998x and 1.08x, none slower with p < 0.05 (the one ratio under 0.95, `mem` on the MI300X at one
thread, 0.86x with p 0.21, has a best-of ratio of 1.19: noise). Small gains of 2 to 5% with p of 0.001 to 0.04 on several (`mfma`
1.03x to 1.04x, `wave` and `atomics` on the RX 6800) are a launch's set-up, where the ids loop is the only change on their path.

Scaling with the final pool, one to eight threads (MI300X, 5 runs, wall time of the device phase, speed-up over one thread):
`mem` 1.96x, 3.81x, 7.43x at 2, 4 and 8; `lds` 1.96x, 3.88x, 7.41x; `divergent` 1.93x, 3.60x, 6.63x; `launch_mid` 1.92x, 3.61x, 6.42x;
`launch_wide` 1.99x, 3.83x, 6.51x; `valu_int` 1.79x, 2.94x, 4.29x (0.336 s at eight threads, the base's time to the millisecond, so it is not this change; probably
the host's other jobs on the sibling threads of CPUs 24-31). No false sharing or shared counter was found: a dispatch's
`DispatchStats` are one for each thread, merged at the end; the allocation table's reader lock spreads threads over 64 slots by a hash
of the thread id (26 distinct slots for 32 threads); the per-thread lookup memory is keyed by the lock's epoch. Past eight threads the host
could not say: 16 CPUs (16-31) gave the same time as 8 for `valu_int` and `mem`, with other jobs on the sibling threads of every core.

### Tests

Both wired into CMake. Mutations were run on scratch copies and each fails the test named.

| test | what it holds | mutations that fail it |
| --- | --- | --- |
| `test_amd_gcn_threads` (`amd/tests/unit`, also run for gfx1100 and gfx1030; the kernels are `vector_add.c` and its LDS reduction) | the pool: n tasks (1 to 40) that wait for one another all run once and at once; the first on the caller; six launching threads do not mix their tasks; a task starts with the launcher's rounding mode and signal mask; a forked child makes its own threads. A dispatch's memory image and counts (waves, instructions, barriers, the class counts and the mix) on 2, 3, 5, 8, 17 and 32 threads equal one thread's, for 1 to 3,000 work-groups of both kernels; 60 dispatches in a row and from four threads at once equal the serial run; a dispatch that fails (a store past its output) throws on 1, 2 and 8 threads and the next one runs | task index off by one; the first task on a new thread; the float environment not applied; the fork child keeping its parent's pool (killed by its alarm); the last work-group of the last range dropped; the first thread's counts not merged; the first error swallowed |
| `test_amd_hip_queue` | order and thread of queued items; `run_inline` on an idle queue (caller's thread, numbered, counted, its failure kept until asked for); an exception is the queue's failure; refused when an item runs or waits or a marker is pending; an item queued while inline work runs waits for it; many threads queueing and waiting, each queued item before the waited-for one that follows it; a wait longer than the spin sleeps and wakes | the sleeper not told to wait for inline work; the idle check dropped; the marker check dropped; the exception not caught; the inline item not numbered; the inline item not marked done. Also clean under ASan, UBSan and TSan |

Also run on the final tree (a full build on server2): every `test_amd_*` unit test (including `test_amd_gcn_same_bits`, which passes
unchanged, `test_amd_gcn_atomics`, `test_amd_gcn_exec`, `test_amd_hip_abi`, `test_amd_rocprofiler`, `test_amd_rocprofiler_args` and the
tests that run for gfx1100, gfx1201, gfx1030 and gfx1250), `test_worker_pool`, `test_memory`, and the AMD end-to-end set: `amd_hip`,
`amd_hipcc`, `amd_graph_host_source`, `amd_hip_abi`, `amd_workgroup`, `amd_hsa`, `amd_hip_on_hsa`, `amd_lost_gpu`, `amd_memtest_patterns`,
`amd_rocprof_*`, `amd_rocblas`, `amd_libraries`, `amd_session`, `amd_debugger`. 202 of 202 pass (`amd_hipsparselt` twice skipped: no
hipSPARSELt). Not run: the PyTorch, vLLM and Ollama sets. Three tests that are not in that list fail on this host with causes that do not involve this change (not re-run on the base: `amd_rocprofiler_abi`: a rocprofiler-sdk header version assertion; `amd_rocprofv3`: `unshare` refused, no user
namespaces; `amd_tools`: the same, and `rocminfo` against ROCm 7.2's).

### Not done, and negative results

* **The NVIDIA engines' worker pool (`vgpu::exec::run_parallel`) for the AMD dispatch, dropped.** The first version of this change
  used it. It is the better pool for a launch with nothing else running, and it is slower than a thread for each task when the CPUs are
  full: `launch_wide` with 8 threads on 8 CPUs 0.64x (MI300X) and 0.70x (RX 6800), p < 0.001 (the launching thread, which it leaves
  waiting and polling, is the ninth thread and takes a CPU from a work-group thread for a time slice, about 200 us a launch);
  `launch_mid` 1.28x (p < 0.001) and 0.94x (p 0.14). `gcn_pool.hpp` runs the first task on the launching thread instead, which
  gave the results above. The shared pool's wait is the same for the NVIDIA engines: worth a look there with the CPUs full.
* **Kernarg alloc and free per launch** (about a third of a `launch`, above): not changed here. A scratch segment moves every later device
  address (the allocator is a bump) and the quarantine of freed ranges, which diagnostics report; making `alloc` and `free` cheaper
  while keeping both means the allocation table's nodes (`live_`, `freed_`, `freed_order_`, the last a map that is only ever popped at
  the front), the lock's epoch and a 64 KiB zeroed chunk for 192 bytes. That is `src/core/memory.cpp` and `brlock.hpp`, not this piece;
  it was done afterwards, keeping every address and diagnostic: "The allocation table and a launch's bookkeeping" at the end.
* **The wave's 64 KiB register file** (14% of `launch` in `set_up_group`, with the ids loop that is gone): the previous round's
  conclusion stands, no bit-safe way to size or recycle it was found.
* **Dynamic scheduling of work-groups across threads**: not done. The ranges are static and contiguous so that groups that wait on later
  ones on their own thread (Stream-K GEMMs) behave as they did; handing out groups one by one changes which groups share a thread.
* **Running small asynchronous copies, fills and event records on the caller when the stream is idle**: not done. It would be faster
  (the hand-off is the cost) but it changes when `hipEventQuery` and `hipStreamQuery` first say yes, and a program may see that.
* **Reusing a stream's thread for the next stream** (`stream` spends 30% in `pthread_create` and thread exit): not done, not
  measured. The queue's thread runs host functions, which may use thread-local state of the runtime.
* **Graph launches**: 12 us for four kernel nodes, 3 us a node, the cost of a plain launch's stream-thread work; the plan is rebuilt
  each launch but is small beside it. Nothing cheap.
* **Event records**: `hipEventRecord` takes `s.mutex` and the event table's mutex and queues an item; no measurement pointed at it.


## The AMD pieces together (branch `speedup5-amdintegrate`)

`speedup` b25ab636 (Tier 1 and 2, amdcorrect, amdmatrix, amdlibs) with `speedup4-amdmemtext`, `speedup4-amdtier` and `speedup4-amdmt` merged in that
order. Every golden file is as recorded (none re-recorded; the base's build still passes them, so they still describe it).

**Where the three clashed, and what was kept.**

* `digest_instruction` / `time_instruction` / `digest_memory_instruction` (`gcn_exec.cpp`, `amd_exec.hpp`): all three probes, the `error` out-parameter
  of the first included. Flavour 4 of `draw` was taken twice: amdmatrix's (every byte 0x3C to 0x41, used by `test_amd_matrix_fast` and
  `amd_matrix_bench`) and the tier's NaN-free floats. The tier's six flavours (4 to 9 on their branch) are now 10 to 15, in `draw`,
  `digest_instruction` and `test_amd_gcn_tier_same_bits`; flavours 0 to 3, which the golden digests use, are untouched.
* `Machine::step`: the tier hook first (`tier_step`), then the lazy `InstCopy` of amdmemtext for VGPR MSBs and VGPR indexing; `Inst::fast` and
  `Inst::mem` both live in `Inst`, and `finalize` fills in both.
* `lds_access`: amdcorrect's `ds_nop` / append / consume guard first, then amdmemtext's `MemFacts` and `lds_fast`.
* Dispatch threads: `pool::run` (amdmt) is the launcher, and it degrades as `start_or_run_here` did: a task the host gives no thread runs on the
  launching thread after the task of its own, in turn, with the bits it would have had (`Pool::Refused::Degrade`). The lockstep launch (tasks that
  wait on one another) uses `pool::run_together` (`Refused::Strict`): every thread is found or made before any task starts, and one the host
  refuses throws `std::system_error` with nothing started, which the dispatch reports as amdcorrect's "the host would not start a thread for the
  dispatch's work-groups: ..." error. `start_or_run_here` is gone. `test_amd_gcn_thread_refused` has a second test for both modes.
* `hip_api.cpp` stream submit: amdmt's `run_inline` and spinning queue, with `no_thread` (hipErrorOutOfMemory) around both `submit` calls.
* The NaN trap again: the merge moved GCC's choice of operand order in `wmma1250`'s scaled products and sums, and 73 gfx1250 WMMA shapes of
  `test_amd_gcn_same_bits` changed their digests. The four operations are pinned (`mfma_mul_o<0>` twice, `mfma_add_o<2>`, `mfma_add_o<0>`), the
  order found by trying all sixteen choices against the recorded digests (one passed).
* `bench-amd`: every workload of every piece is kept; `--counts-via probe` works for all of them; `expected/` gained the five matrix workloads
  that had no entry (recorded with the probe from b25ab636 and equal to the integrated tree's), nothing else changed.

**A/B** (`tools/bench-amd/bench.sh`, 9 runs alternating, one thread and all threads, CPUs 24-31 under `flock /tmp/speedup-bench.lock`, size
full, without counting runs (f81cc0c1 has no probe; the work is identical, `counters identical` is `yes` in every row), server2 load average
1 to 4 before each workload, median ratio B/A, p < 0.001 in every row except `startup`). Columns: the integrated tree against `speedup` f81cc0c1
and against b25ab636, and each of the three pieces' own branch against f81cc0c1 (the figures the pieces reported).

| workload | target | thr | integrated / f81cc0c1 | integrated / b25ab636 | tier alone / f81 | memtext alone / f81 | mt alone / f81 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| valu_int | mi300x | 1 | 8.51 | 8.72 | 8.83 |  |  |
| valu_int | mi300x | all | 8.49 | 8.59 |  |  |  |
| valu_int | rx6800 | 1 | 7.01 | 7.15 | 7.45 |  |  |
| valu_int | rx6800 | all | 7.01 | 7.09 |  |  |  |
| valu_f32 | mi300x | 1 | 9.36 | 9.26 | 9.71 |  |  |
| valu_f32 | mi300x | all | 9.61 | 9.44 |  |  |  |
| valu_f32 | rx6800 | 1 | 6.49 | 6.36 | 6.88 |  |  |
| valu_f32 | rx6800 | all | 6.72 | 6.54 |  |  |  |
| valu_f64 | mi300x | 1 | 4.58 | 4.86 | 4.60 |  |  |
| valu_f64 | mi300x | all | 4.59 | 4.82 |  |  |  |
| valu_f64 | rx6800 | 1 | 5.64 | 5.73 | 5.38 |  |  |
| valu_f64 | rx6800 | all | 5.37 | 5.57 |  |  |  |
| mfma | mi300x | 1 | 2.70 | 1.08 | 1.05 |  |  |
| mfma | mi300x | all | 2.72 | 1.08 |  |  |  |
| mfma_i8 | mi300x | 1 | 4.95 | 0.97 |  |  |  |
| mfma_i8 | mi300x | all | 4.96 | 0.98 |  |  |  |
| mfma_fp8 | mi300x | 1 | 7.45 | 1.04 |  |  |  |
| mfma_fp8 | mi300x | all | 7.45 | 1.04 |  |  |  |
| mfma_bf16 | mi300x | 1 | 3.37 | 1.06 |  |  |  |
| mfma_bf16 | mi300x | all | 3.36 | 1.06 |  |  |  |
| mfma_f64 | mi300x | 1 | 3.73 | 1.12 |  |  |  |
| mfma_f64 | mi300x | all | 3.25 | 1.11 |  |  |  |
| smfmac | mi300x | 1 | 3.00 | 1.03 |  |  |  |
| smfmac | mi300x | all | 3.01 | 1.02 |  |  |  |
| mem | mi300x | 1 | 3.08 | 3.10 | 2.99 | 1.10 |  |
| mem | mi300x | all | 3.09 | 3.12 |  |  |  |
| mem | rx6800 | 1 | 3.28 | 3.14 | 3.28 | 1.13 |  |
| mem | rx6800 | all | 3.27 | 3.17 |  |  |  |
| lds | mi300x | 1 | 3.79 | 3.71 | 3.66 | 2.13 |  |
| lds | mi300x | all | 3.87 | 3.86 |  |  |  |
| lds | rx6800 | 1 | 3.30 | 3.31 | 3.12 | 1.90 |  |
| lds | rx6800 | all | 3.30 | 3.33 |  |  |  |
| narrow | mi300x | 1 | 1.94 | 1.98 | 1.91 | 1.11 |  |
| narrow | mi300x | all | 1.95 | 1.99 |  |  |  |
| narrow | rx6800 | 1 | 2.10 | 2.05 | 2.11 | 1.10 |  |
| narrow | rx6800 | all | 2.08 | 2.06 |  |  |  |
| smem | mi300x | 1 | 3.60 | 3.81 | 3.30 | 1.11 |  |
| smem | mi300x | all | 3.64 | 3.85 |  |  |  |
| smem | rx6800 | 1 | 3.12 | 3.17 | 2.81 | 1.18 |  |
| smem | rx6800 | all | 3.15 | 3.23 |  |  |  |
| buf | mi300x | 1 | 1.97 | 1.92 | 1.39 | 1.32 |  |
| buf | mi300x | all | 1.98 | 1.93 |  |  |  |
| buf | rx6800 | 1 | 2.06 | 2.02 | 1.45 | 1.32 |  |
| buf | rx6800 | all | 2.07 | 2.03 |  |  |  |
| ldsx | mi300x | 1 | 6.96 | 7.00 | 2.70 | 2.28 |  |
| ldsx | mi300x | all | 7.39 | 7.37 |  |  |  |
| ldsx | rx6800 | 1 | 6.32 | 6.17 | 2.83 | 2.02 |  |
| ldsx | rx6800 | all | 6.46 | 6.40 |  |  |  |
| flat | mi300x | 1 | 3.35 | 3.29 | 2.64 | 1.13 |  |
| flat | mi300x | all | 3.27 | 3.31 |  |  |  |
| flat | rx6800 | 1 | 3.50 | 3.55 | 2.78 | 1.13 |  |
| flat | rx6800 | all | 3.53 | 3.54 |  |  |  |
| atomics | mi300x | 1 | 2.05 | 2.08 | 1.85 | 1.15 |  |
| atomics | mi300x | all | 1.96 | 1.91 |  |  |  |
| atomics | rx6800 | 1 | 2.23 | 2.25 | 1.98 | 1.19 |  |
| atomics | rx6800 | all | 2.06 | 1.97 |  |  |  |
| divergent | mi300x | 1 | 4.40 | 4.47 | 4.53 |  |  |
| divergent | mi300x | all | 4.41 | 4.48 |  |  |  |
| divergent | rx6800 | 1 | 4.30 | 4.21 | 4.32 |  |  |
| divergent | rx6800 | all | 4.24 | 4.27 |  |  |  |
| wave | mi300x | 1 | 11.12 | 11.03 | 11.41 |  |  |
| wave | mi300x | all | 10.92 | 10.75 |  |  |  |
| wave | rx6800 | 1 | 8.84 | 8.79 | 9.12 |  |  |
| wave | rx6800 | all | 8.74 | 8.66 |  |  |  |
| launch | mi300x | 1 | 1.08 | 1.14 |  |  | 1.05 |
| launch | mi300x | all | 1.08 | 1.14 |  |  | 0.99 |
| launch | rx6800 | 1 | 1.14 | 1.15 |  |  | 1.04 |
| launch | rx6800 | all | 1.14 | 1.14 |  |  | 1.04 |
| api | mi300x | 1 | 3.27 | 3.23 |  |  | 3.50 |
| api | mi300x | all | 3.34 | 3.44 |  |  | 3.49 |
| api | rx6800 | 1 | 3.45 | 3.30 |  |  | 3.44 |
| api | rx6800 | all | 3.30 | 3.27 |  |  | 3.31 |
| apicheap | mi300x | 1 | 1.09 | 1.10 |  |  | 1.10 |
| apicheap | mi300x | all | 1.10 | 1.10 |  |  | 1.11 |
| apicheap | rx6800 | 1 | 1.11 | 1.09 |  |  | 1.11 |
| apicheap | rx6800 | all | 1.11 | 1.09 |  |  | 1.12 |
| launch_wide | mi300x | 1 | 2.31 | 2.33 |  |  | 1.04 |
| launch_wide | mi300x | all | 2.66 | 2.70 |  |  | 1.14 |
| launch_wide | rx6800 | 1 | 2.09 | 2.06 |  |  | 1.01 |
| launch_wide | rx6800 | all | 1.42 | 1.62 |  |  | 1.01 |
| launch_mid | mi300x | 1 | 2.31 | 2.33 |  |  | 1.05 |
| launch_mid | mi300x | all | 4.86 | 4.85 |  |  | 2.46 |
| launch_mid | rx6800 | 1 | 2.08 | 2.06 |  |  | 1.02 |
| launch_mid | rx6800 | all | 3.57 | 3.50 |  |  | 1.93 |
| launch_sync | mi300x | 1 | 1.61 | 1.59 |  |  | 1.22 |
| launch_sync | mi300x | all | 1.62 | 1.62 |  |  | 1.23 |
| launch_sync | rx6800 | 1 | 1.55 | 1.54 |  |  | 1.16 |
| launch_sync | rx6800 | all | 1.56 | 1.55 |  |  | 1.16 |
| memcpy | mi300x | 1 | 2.11 | 2.16 |  |  | 2.17 |
| memcpy | mi300x | all | 2.13 | 2.18 |  |  | 2.17 |
| memcpy | rx6800 | 1 | 2.12 | 2.15 |  |  | 2.11 |
| memcpy | rx6800 | all | 2.13 | 2.12 |  |  | 2.15 |
| stream | mi300x | 1 | 1.21 | 1.22 |  |  | 1.16 |
| stream | mi300x | all | 1.22 | 1.21 |  |  | 1.18 |
| stream | rx6800 | 1 | 1.29 | 1.47 |  |  | 1.20 |
| stream | rx6800 | all | 1.26 | 1.45 |  |  | 1.20 |
| graph | mi300x | 1 | 1.36 | 1.36 |  |  | 1.04 |
| graph | mi300x | all | 1.38 | 1.36 |  |  | 1.04 |
| graph | rx6800 | 1 | 1.33 | 1.34 |  |  | 1.03 |
| graph | rx6800 | all | 1.33 | 1.33 |  |  | 1.04 |
| startup | mi300x | 1 | 1.04 | 0.98 |  |  |  |
| startup | mi300x | all | 1.02 | 1.00 |  |  |  |
| startup | rx6800 | 1 | 1.01 | 1.00 |  |  |  |
| startup | rx6800 | all | 1.01 | 1.01 |  |  |  |
| wmma | rx7900xtx | 1 | 25.12 | | | | |
| wmma | rx7900xtx | all | 25.38 | | | | |
| wmma | rx9070xt | 1 | 25.30 | | | | |
| wmma | rx9070xt | all | 25.63 | | | | |
| wmma_iu8 | rx7900xtx | 1 | 9.24 | | | | |
| wmma_iu8 | rx7900xtx | all | 9.50 | | | | |
| wmma_iu8 | rx9070xt | 1 | 7.16 | | | | |
| wmma_iu8 | rx9070xt | all | 7.34 | | | | |

(`startup` is unchanged, 0.98 to 1.05. `wmma` and `wmma_iu8` on the simulated RX 7900 XTX and RX 9070 XT: 25.1 to 25.6 x and 7.2 to 9.5 x f81cc0c1.
`v_pk_fma_f16` costs 200 to 260 ns an execution in b25ab636 and in the integrated tree alike (`amd_matrix_bench`; f81cc0c1's was 3103 ns), a
`v_mfma_f32_16x16x16_f16` 1.6 us in both: the tier takes no packed 16-bit or matrix instruction, so those gains are untouched.)

**What the claims became.** Tier: 4.6 to 11.1 x on `valu_*` / `wave` / `divergent` (alone 4.6 to 11.4 x; at most 6 % less, rx6800 `valu_f32`
and `valu_int`). Mt: `api` 3.3 to 3.4 x (alone 3.3 to 3.5 x), `memcpy` 2.1 x (2.1 to 2.2 x), `launch_mid` with all threads 4.9 / 3.6 x
(alone 2.5 / 1.9 x), no row below 0.93 of the piece alone. Matrix: the `mfma*` rows are 2.7 to 7.4 x and `wmma` 25 x, the same as b25ab636's
(`mfma` is 1.08 x faster than b25ab636 for the tier's scalar and branch instructions). Libs: unchanged, as above.
**One piece's gain shrank: amdmemtext's, on the instructions the tier now runs itself.** Its contribution on top of the tier (integrated / tier
alone) against what it gave alone: `mem` 1.00 to 1.03 x (alone 1.10 to 1.13), `lds` 1.04 to 1.06 x (alone 1.9 to 2.1), `narrow` 1.00 to 1.02 x
(alone 1.10), `atomics` 1.11 to 1.13 x (alone 1.15 to 1.19); not shrunk where the tier does not reach: `ldsx` 2.2 to 2.6 x (alone 2.0 to 2.3),
`buf` 1.42 x (1.32), `flat` 1.26 x (1.13), `smem` 1.09 to 1.11 x (1.11 to 1.18). The memtext code is still what runs the instructions the tier hands
back (a wave with VGPR MSBs or indexing, MODE not at its default, the debugger or tracing, an access that is not plain, wide LDS forms) and
the interpreter's own steps of the other workloads.

**Tests** (server2, Release; each with `VGPU_AMD_TIER=1` and `=0`): `ctest -R "amd|gcn|rocprof|hipcc|hip_|memrmw|memruns|mem_facts|mfma|packed|matrix|tier|bench_amd"`,
251 tests, 3 failed in both modes (the rest passed or were skipped), the same three as in b25ab636 (`amd_rocprofiler_abi`: the host's rocprofiler-sdk is another
version; `amd_rocprofv3` and `amd_tools`: unprivileged user namespaces and `rocminfo` do not work on server2); the rest skipped for want of
ROCm libraries or cards. Among them `test_amd_gcn_same_bits`, `_packed_same_bits`, `_memory_same_bits`, `_tier_same_bits`,
`test_amd_codeobject_same_bits`, `test_amd_matrix_fast`, `test_amd_gcn_threads*`, `test_amd_hip_queue`, `test_amd_gcn_thread_refused`,
`test_amd_hip_thread_refused`, `bench_amd_counters`. The NVIDIA tests that share code (`test_memory`, `test_memrmw*`, `test_worker_pool`,
`test_fastpath`, `test_exec*`): 8 of 8.


## The speedup branch merged with origin/main (#382, #400) (branch `speedup6-main`)

`speedup` bb46caaf with origin/main ab904cbc: "AMD: gfx942/gfx950 guide sweep, gfx90a images, hostcall, PC sampling, RVS, vLLM graphs"
(#382, which changed the behaviour of many instructions and edited `gcn_exec.cpp` inside the old if-ladders) and #400 (cuTensorNet).

**How the 32 conflicts of `gcn_exec.cpp` were merged.** `amd/tools/ladder-to-switch.py` is a deterministic rewrite, so the file was merged three
ways *after* it: the merge base's ladders converted, origin/main's converted, and `speedup`'s file (already converted), `git merge-file`
over the three. That took the 32 hunks down to 21, all real: the decoded memory facts of amdmemtext against #382's text-based memory code, the
tile MFMA code of amdmatrix against #382's MFMA changes, the SOPP switch against new SOPP rungs, `Machine::step`, and the packed and conversion
paths. Each was written by hand keeping both sides (and the build script's two `for src in` lists were joined).

**The golden digests were re-recorded, from origin/main plus the probe only.** The arbiter is origin/main ab904cbc with the probe functions
(`digest_instruction`, `digest_memory_instruction`, `time_instruction`, and stubs of `set_matrix_fast_path`) added and nothing else
(the same four test programs, built there). Against the committed digests, run on the *old* corpora (so that every seed and position
is the same), the arbiter differs from them in 1,044 of 16,304 `same_bits` shapes, 336 of 2,143 `packed_same_bits` shapes and 66 of
15,566 `memory_same_bits` shapes; every one is a thing #382 changed on purpose:

| family (shapes changed) | what #382 did |
| --- | --- |
| MFMA and SMFMAC, 639 | CBSZ/ABID checks and BLGP lane permutations, the f8f6f4 K layout, denormal flush of float inputs, sparse products skipping pruned places, the lane layout of several-block forms |
| `v_div_scale`, `v_div_fmas`, `v_div_fixup` f16/f32/f64, 193 | the division sequence of the guides' pseudocode (`gcn_divide.hpp`) |
| SOPP corpus, 60 | `s_setkill`, `s_sethalt`, `s_set_gpr_idx_mode`, `s_cbranch_cdbg*`, `s_incperflevel`, VSKIP, ... |
| memory, 43 + 66 | 64-bit and `cond_sub`/`sub_clamp` atomics, ADDTID, buffer loads into LDS, scalar buffer stores and atomics |
| `v_dot2*` 10 + 237, `v_cvt_f32_f16` 40 + 39, `v_sin`/`v_cos` 4 + 60, min/max, `v_fma_f16`, omod | the guides' denormal flush, `abs` of a NaN, `v_sin`/`v_cos` in turns, omod ignored by MODE |
| scalar, 27 | `s_movrel*`, `s_round_mode`, `s_max` SCC on a tie, ... |

#382 also grew the corpora (`isa_corpus.txt`, `_gfx90a`, `_gfx950`, `_images_gfx90a`: 16,806 shapes now, 502 more), and the seed of a
shape is its position, so `same_bits_digests.txt` and `mem_same_bits_digests.txt` are re-recorded whole from the arbiter, `packed_same_bits_digests.txt`
whole as well (its corpus did not change; 336 digests did), and `codeobject_digests.txt` from the arbiter with the six entries of `speedup`'s own
code objects kept (one hipcc object's digest differs: #382 rebuilt it). The merged tree passes all of them as they are committed; the arbiter
build, the old-corpora run and the recorded files are the evidence, and no digest was taken from this tree.

**What the merge itself broke, found by those digests and the tests**, each fixed without touching a golden file:
* GCC's operand order for two NaNs or two zeros moved (the trap of every round): `wmma1250`'s scaled products and sums, the packed min and max
  on RDNA and gfx90a, `v_dot2_f32_f16` and `v_dot2c_f32_f16` (one order for the guides' flushing path, another for the rest) are pinned (`first_mul`,
  `first_add`, `pk_fminf`, `pk_fmaxf`); `v_maxmin_f32` and its kin call libm's `fminf`/`fmaxf` explicitly, as the recorded build did.
* `v_cvt_f32_f16` with `|x|`: `lane_half_f` cleared the sign of a NaN by comparison; #382 clears the bit.
* `gfx9_added_alu` renames an instruction (`v_mad_legacy_u16` to `v_mad_u16`): the copy is now `finalize`d (the hash and the flags are of the name).
* The matrix tile path hands to the reference whatever #382 changed in the reference: the small-float and scaled forms, CBSZ past the blocks, BLGP,
  several blocks with several groups of K a lane; a sparse tile whose B has a NaN or infinity, a float tile with a denormal input (flush); EXEC is
  ignored, as it is on the card. (`v_mfma_*_f8f6f4` and the scaled forms run at the reference's speed until the tile code learns the new K layout.)
* The tier: no handler where gfx942/gfx950 apply or ignore an output modifier or a clamp, `s_max_*` SCC on a tie has its own handlers there, `v_lshl_add_u64`
  has a guide-architecture one (count over 4 is zero), and no tier while PC sampling is attached or VSKIP is set.
* `dot2_f16_wave` (amdlibs' whole-wave dot product) is not used on gfx942/gfx950, whose dot products flush denormals.
* A tbuffer instruction through the private segment buffer stays refused, as in origin/main and the base (amdmemtext's decoded facts would have run it;
  the corpora now have the shape).
* `ds_append`/`ds_consume` run (#382 models them); amdcorrect's "not modelled" guard lets them through, and its test says so. The decoded memory
  facts know the new names (`cond_sub`, `sub_clamp`, ADDTID, `s_buffer_*`, the volatile cache forms).

**Tests** (server2, Release, each with `VGPU_AMD_TIER=1` and `=0`): `ctest -R "amd|gcn|rocprof|hipcc|hip_|memrmw|memruns|mem_facts|mfma|packed|matrix|tier|bench_amd|hostcall|pcsampl|rvs|vllm|graph|pytorch"`,
310 tests; the same three fail as on `speedup` and on origin/main (`amd_rocprofiler_abi`, `amd_rocprofv3`, `amd_tools`: this host's ROCm and user
namespaces); the vLLM, PyTorch and `amd_rvs` ones are skipped for want of the packages. #382's own (`test_amd_gcn_cdna_gaps`, `test_amd_gcn_divide`,
`test_amd_hostcall`, `test_amd_rocprofiler_pcsampling*`, and the new programs of `amd_hipcc`) pass. #400's and the shared NVIDIA ones
(`e2e_cutensornet_*`, `e2e_ipc`, `test_memory`, `test_memrmw*`, `test_worker_pool`, `test_fastpath`, `test_exec*`, `e2e_sass_tier`): 24 of 24.

**A/B** (`tools/bench-amd`, 9 runs alternating, 1 thread and all threads, CPUs 24-31 under `flock`, full size, `--no-counts` because origin/main has no probe;
`counters identical: yes` in every row; p < 0.001 except `startup`): the merged tree against a build of origin/main ab904cbc, and (one thread) against
`speedup` bb46caaf, to see what the merge cost. Ratio of the medians, > 1 is faster.

| workload | target | thr | merged / origin/main ab904cbc | merged / speedup bb46caaf (1 thread) |
| --- | --- | ---: | ---: | ---: |
| valu_int | mi300x | 1 | 20.22 | 0.99 |
| valu_int | mi300x | all | 20.38 |  |
| valu_int | rx6800 | 1 | 18.04 | 1.00 |
| valu_int | rx6800 | all | 17.87 |  |
| valu_f32 | mi300x | 1 | 21.32 | 1.04 |
| valu_f32 | mi300x | all | 21.08 |  |
| valu_f32 | rx6800 | 1 | 16.65 | 0.99 |
| valu_f32 | rx6800 | all | 17.52 |  |
| valu_f64 | mi300x | 1 | 9.30 | 0.91 |
| valu_f64 | mi300x | all | 9.37 |  |
| valu_f64 | rx6800 | 1 | 13.89 | 0.95 |
| valu_f64 | rx6800 | all | 13.89 |  |
| mfma | mi300x | 1 | 7.53 | 0.95 |
| mfma | mi300x | all | 7.58 |  |
| mfma_i8 | mi300x | 1 | 7.39 | 1.00 |
| mfma_i8 | mi300x | all | 7.53 |  |
| mfma_fp8 | mi300x | 1 | 9.12 | 0.94 |
| mfma_fp8 | mi300x | all | 9.09 |  |
| mfma_bf16 | mi300x | 1 | 6.22 | 0.94 |
| mfma_bf16 | mi300x | all | 6.29 |  |
| mfma_f64 | mi300x | 1 | 5.04 | 0.99 |
| mfma_f64 | mi300x | all | 4.59 |  |
| smfmac | mi300x | 1 | 7.11 | 0.89 |
| smfmac | mi300x | all | 7.26 |  |
| mem | mi300x | 1 | 5.85 | 0.98 |
| mem | mi300x | all | 5.88 |  |
| mem | rx6800 | 1 | 7.14 | 0.99 |
| mem | rx6800 | all | 7.21 |  |
| lds | mi300x | 1 | 11.31 | 0.98 |
| lds | mi300x | all | 11.73 |  |
| lds | rx6800 | 1 | 9.45 | 0.94 |
| lds | rx6800 | all | 9.60 |  |
| narrow | mi300x | 1 | 2.92 | 0.96 |
| narrow | mi300x | all | 3.04 |  |
| narrow | rx6800 | 1 | 3.80 | 0.99 |
| narrow | rx6800 | all | 3.85 |  |
| smem | mi300x | 1 | 8.69 | 0.89 |
| smem | mi300x | all | 8.88 |  |
| smem | rx6800 | 1 | 8.69 | 0.81 |
| smem | rx6800 | all | 8.73 |  |
| buf | mi300x | 1 | 2.69 | 1.01 |
| buf | mi300x | all | 2.76 |  |
| buf | rx6800 | 1 | 3.27 | 1.00 |
| buf | rx6800 | all | 3.30 |  |
| ldsx | mi300x | 1 | 27.56 | 0.95 |
| ldsx | mi300x | all | 28.99 |  |
| ldsx | rx6800 | 1 | 22.95 | 0.95 |
| ldsx | rx6800 | all | 23.79 |  |
| flat | mi300x | 1 | 6.67 | 0.95 |
| flat | mi300x | all | 6.70 |  |
| flat | rx6800 | 1 | 7.85 | 0.92 |
| flat | rx6800 | all | 7.85 |  |
| atomics | mi300x | 1 | 4.28 | 0.98 |
| atomics | mi300x | all | 24.35 |  |
| atomics | rx6800 | 1 | 5.86 | 1.01 |
| atomics | rx6800 | all | 16.89 |  |
| divergent | mi300x | 1 | 12.94 | 1.00 |
| divergent | mi300x | all | 12.87 |  |
| divergent | rx6800 | 1 | 13.89 | 1.00 |
| divergent | rx6800 | all | 13.88 |  |
| wave | mi300x | 1 | 18.76 | 0.97 |
| wave | mi300x | all | 18.60 |  |
| wave | rx6800 | 1 | 16.99 | 1.00 |
| wave | rx6800 | all | 16.85 |  |
| launch | mi300x | 1 | 1.44 | 1.00 |
| launch | mi300x | all | 1.44 |  |
| launch | rx6800 | 1 | 1.48 | 1.00 |
| launch | rx6800 | all | 1.47 |  |
| api | mi300x | 1 | 3.21 | 1.00 |
| api | mi300x | all | 3.35 |  |
| api | rx6800 | 1 | 3.33 | 1.01 |
| api | rx6800 | all | 3.34 |  |
| apicheap | mi300x | 1 | 1.11 | 0.98 |
| apicheap | mi300x | all | 1.09 |  |
| apicheap | rx6800 | 1 | 1.11 | 0.99 |
| apicheap | rx6800 | all | 1.10 |  |
| launch_wide | mi300x | 1 | 4.27 | 1.01 |
| launch_wide | mi300x | all | 3.25 |  |
| launch_wide | rx6800 | 1 | 4.08 | 1.02 |
| launch_wide | rx6800 | all | 3.17 |  |
| launch_mid | mi300x | 1 | 4.26 | 1.02 |
| launch_mid | mi300x | all | 6.31 |  |
| launch_mid | rx6800 | 1 | 4.09 | 1.01 |
| launch_mid | rx6800 | all | 5.46 |  |
| launch_sync | mi300x | 1 | 2.15 | 1.00 |
| launch_sync | mi300x | all | 2.20 |  |
| launch_sync | rx6800 | 1 | 2.10 | 1.00 |
| launch_sync | rx6800 | all | 2.10 |  |
| memcpy | mi300x | 1 | 2.17 | 1.01 |
| memcpy | mi300x | all | 2.18 |  |
| memcpy | rx6800 | 1 | 2.18 | 0.99 |
| memcpy | rx6800 | all | 2.19 |  |
| stream | mi300x | 1 | 1.47 | 0.99 |
| stream | mi300x | all | 1.44 |  |
| stream | rx6800 | 1 | 1.63 | 0.98 |
| stream | rx6800 | all | 1.55 |  |
| graph | mi300x | 1 | 1.91 | 1.01 |
| graph | mi300x | all | 1.91 |  |
| graph | rx6800 | 1 | 1.86 | 1.01 |
| graph | rx6800 | all | 1.87 |  |
| startup | mi300x | 1 | 1.06 | 1.09 |
| startup | mi300x | all | 1.07 |  |
| startup | rx6800 | 1 | 1.02 | 0.99 |
| startup | rx6800 | all | 1.02 |  |
| wmma | rx7900xtx | 1 | 26.52 | |
| wmma | rx7900xtx | all | 26.82 | |
| wmma | rx9070xt | 1 | 27.66 | |
| wmma | rx9070xt | all | 28.24 | |
| wmma_iu8 | rx7900xtx | 1 | 10.34 | |
| wmma_iu8 | rx7900xtx | all | 10.52 | |
| wmma_iu8 | rx9070xt | 1 | 7.46 | |
| wmma_iu8 | rx9070xt | all | 7.77 | |

Against origin/main every workload is faster (startup within noise); #382's guide sweep made main slower than the `speedup` pieces' baseline f81cc0c1
on the gfx942 workloads (the ratios are larger than the 2-11 x of the pieces). Against `speedup` before the merge, the gfx942 and RX 6800 rows
lost 3 to 19 % (`smem` 11 % and 19 %, `smfmac` 11 %, `valu_f64` on the MI300X 9 %, `flat` on the RX 6800 8 %, the MFMA rows 5 to 6 %,
`lds` and `ldsx` 5 to 6 %): the checks #382 added to the interpreter's `step` (PC sampling, VSKIP), the denormal flushes and the omod/clamp rules of the guides, and
the tile path's declines. Not profiled further. The first run of this A/B had `mem` and `flat` on the MI300X at 0.46 x and 0.48 x and
`launch_wide`/`launch_mid` at 0.68 x: the tier had been told not to run `v_lshl_add_u64` on gfx942, which the address arithmetic of those kernels uses;
it has a handler now.

## The AMD merge's cost, found and won back (branch `sn-amdmerge`)

The merge of #382 into `speedup` (above) left the gfx942 and RX 6800 rows 3 to 19 % slower than `speedup` before it (`bb46caaf`), "not profiled
further". Profiled now, with the SIGPROF sampler (`tools/bench-amd/profile/`, a `-g` build of each tree, `SIGPROF_STACK=1`), instruction counts and the
set of instructions that run in the interpreter (`VGPU_AMD_TIER_STATS=1`) identical in both trees on every bench workload (so the tier's declines, the
omod/clamp rules and the denormal flushes are not the cause: nothing the benchmarks run was handed back), the causes were three:

* **`scalar_alu` was an if-ladder of 81 rungs again.** The interpreter's `op == "name"_op` ladders are switches on the name's hash
  (`amd/tools/ladder-to-switch.py`); #382 edited inside the old ladders, and the merge kept its ladder form for `scalar_alu`. One rung of it
  (`s_flbit_i32_i64`, #382's) had a `break` inside a `for`, which is outside a loop of the converter's own: the converter leaves a chain alone that has one (a switch would
  capture the `break`), and said nothing (`chains: 0`). Every scalar instruction the tier does not do (`s_mul_hi_u32`, `s_cselect`, `s_lshr_b32` ...) walked down up
  to 81 hash comparisons: 9 % of the RX 6800 `smem` kernel's samples (`OpName::operator==`). The rung is written without the `break`, the converter run, and
  `lint_amd_ladders` (`tests/lint/check_ladders.py`) fails for any chain of 12 or more instruction names left as an if-ladder in `gcn_exec.cpp`
  (on the file before this change it fails with `gcn_exec.cpp:1484: an if-ladder of 81 rungs naming 111 instructions`).
* **`scalar_load` grew a frame.** #382 added the buffer stores and buffer atomics to it, and the plain dword load (`s_load_dword*`, the case that matters) paid
  for the function's size. The plain load is the function now (`scalar_load`, address in `scalar_load_base`) and the rest is `scalar_load_general`, out of line.
* **`Machine::step` was one big function.** Its frame (0x6f8 bytes, a stack protector, six callee-saved registers pushed) was paid for by every
  instruction, the tier's included, and #382's additions (PC sampling, VSKIP) made it bigger. `step` is now the debugger check (once per machine, `debug_on`,
  instead of a call to `debug::active()` per instruction; the session's check is `debug_check`, out of line), the fetch and the tier; the interpreter's own work is
  `step_interp`, out of line. This is where the rows beyond the pre-merge numbers come from.

Alternating A/B, `tools/bench-amd/bench.sh --build pre --build work`, 11 runs, size full, 1 thread, pinned to CPUs 20-27 of server2 (load 14 to 26: other
agents' builds), median ratio of the rate (> 1: faster); the A/A noise on the same rows (two copies of one build) is 0.2 to 1 %. `pre` is `bb46caaf`;
"merged" is the ratio the merge had (the table above); counters identical in every row.

| workload | target | merged / pre | now / pre |
| --- | --- | ---: | ---: |
| valu_int | mi300x | | 1.22 |
| valu_int | rx6800 | | 1.21 |
| valu_f32 | mi300x | | 1.13 |
| valu_f32 | rx6800 | | 1.05 |
| valu_f64 | mi300x | 0.91 | 1.08 |
| valu_f64 | rx6800 | 0.95 | 1.10 |
| mfma | mi300x | 0.95 | 1.05 |
| mfma_i8 | mi300x | 1.00 | 1.07 |
| mfma_fp8 | mi300x | 0.94 | 0.89 |
| mfma_bf16 | mi300x | 0.94 | 1.12 |
| mfma_f64 | mi300x | 0.99 | 1.00 |
| smfmac | mi300x | 0.89 | 0.93 |
| mem | mi300x | 0.98 | 1.05 |
| mem | rx6800 | 0.99 | 1.08 |
| lds | mi300x | 0.98 | 1.09 |
| lds | rx6800 | 0.94 | 1.13 |
| narrow | mi300x | 0.96 | 0.98 |
| narrow | rx6800 | 0.99 | 1.04 |
| smem | mi300x | 0.89 | 1.22 |
| smem | rx6800 | 0.81 | 1.27 |
| buf | mi300x | 1.01 | 1.03 |
| buf | rx6800 | 1.00 | 1.04 |
| ldsx | mi300x | 0.95 | 1.12 |
| ldsx | rx6800 | 0.95 | 1.19 |
| flat | mi300x | 0.95 | 1.04 |
| flat | rx6800 | 0.92 | 1.05 |
| atomics | mi300x | 0.98 | 1.05 |
| atomics | rx6800 | 1.01 | 1.06 |
| divergent | mi300x | 1.00 | 1.21 |
| divergent | rx6800 | 1.00 | 1.27 |
| wave | mi300x | 0.97 | 1.20 |
| wave | rx6800 | 1.00 | 1.26 |

The `scalar_alu` switch alone (before the other two) took `smem` from 0.88 and 0.84 of `pre` to 0.96 and 0.98, `lds` on the RX 6800 5 % up; the other two
pieces gave the rest. Left: `mfma_fp8` (0.89) and `smfmac` (0.93). Profiled, the matrix tile path's code is the same in both trees and the
samples are on the same lines; the extra time is in the small-float table lookup of the decode (`matrix_elem<'e'>`: 61 samples of `pre`, 63 of the merged tree, 100 of the tree with
the split `step`), and by direct runs the merged tree was 0.97 of `pre` on `mfma_fp8` and 0.91 on `smfmac`, the split one 0.89 and 0.93. Code that is the same
running at different speeds with the layout of the code and the heap points at an effect of addresses (a store-to-load false dependency between the tile buffers and the
table is a guess, not measured); it is left for the matrix tile path's rewrite, not a fix here.
The same-bits guards pass unchanged in Release (`test_amd_gcn_same_bits`, `_packed_`, `_memory_`, `_tier_`, `test_amd_codeobject_same_bits`, `test_amd_matrix_fast`,
`test_amd_mfma_math`, `test_amd_nan_order`, each also at `VGPU_SIMD=base`).

## A scratch access that wrapped its range check (branch `speedup7-scratch`)

`scratch_at` tested `offset + bytes > scratch_per_lane` with an unsigned offset. An instruction offset that is negative (`scratch_store_b128 v9,
v[1:4], off offset:-16`) with a register offset of zero is a number just under 2^64; adding the size wrapped the sum to zero and passed
the test, and the access landed before the work-item's scratch, on the heap (amdmemtext found it as a crash of the memory digest probe, which
skipped such shapes). The test is now `offset > per_lane || bytes > per_lane - offset`, and the access is refused with the error every other
access past the work-item's scratch gets. The other range checks were read and left: flat accesses to the private aperture take their offset
from an address that is at least the aperture's base (or 52 bits), the LDS ones from the shared aperture or a 32-bit field, so none of them wraps;
`scratch_buffer_access`, `scratch_load_lds` and gfx1250's signed sum go through `scratch_at`.

The golden digests changed only for the shapes this changes, recorded from origin/main plus the probe plus only this fix (the same arbiter as the merge's):
3 shapes of `same_bits_digests.txt` (gfx1100 `scratch_load_b128 v[5:8], v9, off offset:-16`, `scratch_store_b128 v9, v[1:4], off offset:-16`,
`scratch_store_b64 v1, v[2:3], off offset:-8`) and 28 of `mem_same_bits_digests.txt`: the 28 negative-offset scratch shapes of gfx1100 (7), gfx1201 (4) and
gfx1250 (17), which the probe used to skip (the skip is gone; their digest was 0). `test_amd_gcn_scratch_negative` runs 12 stores and 12 loads of each of
gfx942, gfx1030 (global accesses with the segment bits set to scratch), gfx1100 and gfx1250 at minus their size from zero registers (digest flavour 16, all zero)
and wants the refusal; with the check as it was, it fails (`gave []`).

## One PTX atomic read-modify-write (branch `speedup8-ci`)

`speedup` and origin/main (#402, round 10) each had made a PTX device atomic one table lookup instead of two: `MemoryManager::rmw_scalar` with
the interpreter's `rmw_global`, and `rmw_begin` / `rmw_commit` with the interpreter's inline fast path. Main's is the one kept (it covers the same
cases, declines in the same ones, and has its own mutation-checked test, `test_atom_rmw`); `rmw_scalar_fn`, `rmw_scalar` and `rmw_global` are gone, and
`test_memrmw` makes its one-lookup read-modify-writes with `rmw_begin` / `rmw_commit`. `atomic_rmw` (the lock-free compare-and-swap form the SASS and AMD
engines use) and `load_fields` stay.

## What PR #403's CI found (branch `speedup8-ci`)

* **`test_amd_gcn_same_bits` on the CUDA 12.4 job, 11 of 16,806 shapes** (`v_add_f16` / `v_mul_f16` and their sdwa, e64 and dpp forms): the job builds with GCC 12
  (this machine's is 13). Reproduced here with GCC 12.4's packages unpacked in a directory (`apt-get download` and `dpkg -x`, no root). A half add
  or multiply is done in float, and with a NaN in both operands the result keeps one operand's payload; GCC 12 made the first operand that one, GCC 13
  the second (the recorded digests are the second's). `half_add`, `half_sub` and `half_mul` spell the order out (`first_addf` / `first_mulf` in
  `gcn_exec.cpp`). All the digest tests pass with both compilers, and with `VGPU_SIMD=base` (the AMD engine's AVX2/FMA/F16C paths now obey it too, and the
  digest tests are registered a second time with it: `*_simd_base`).
* **The sanitizer jobs (Debug, no optimization)**: signed overflow in the integer reference code (`v_mad_i32_i24`, the WMMA/MFMA integer sums, which the
  hardware wraps) is now done unsigned (`wrap_madd`), with the same results. `test_amd_mfma_math` compared a NaN-carrying sum of the fixed operand
  order with the compiler's own order in the test; it compares a NaN as a NaN now. In an unoptimized build 25 to 82 of the 16,000 shapes of the digest tests have other NaN payloads
  (or, for `v_cos_f32` and a few atomics, other results; those were not traced further: the same shapes pass in every optimized build and with the tier off or on),
  so the three digest tests allow 2 % of the shapes to differ in a build without optimization, with a note, and none in an optimized one.
* **ThreadSanitizer**: no data race was reported anywhere. `test_amd_gcn_threads` died in `a_forked_child_launches_on_threads_of_its_own`: TSan does not
  support threads started in a forked child of a threaded process. That case is skipped under TSan.
* **Lint**: the `interp` job of `sass-ptx.yml` has the pr-guard step; `test_mma_fast.cpp` had a narrowing conversion; the whole tree builds with `-DVGPU_WERROR=ON`
  (a missing `default:` in `global_access`'s switch and a narrowing in the probe were ours).

## Sanitizer fixes (round 11) (branch `speedup-merge`, pushed to `speedup`)

PR #403's sanitizer jobs failed on its own AMD code. Reproduced on the rig (c7i.4xlarge, the CI image `cuda12.0-9f7f0f8400f7`, CUDA 12.0 and gcc 12) with the
CI recipe: `-DCMAKE_BUILD_TYPE=Debug -fsanitize=address,undefined` and `-fsanitize=thread -O1`. Both jobs' failures were in the AMD interpreter
and its tests. Three kinds of cause, none a race and none a timeout.

**1. Signed integer overflow (undefined behaviour; the three UBSan lines of the ASan job).** The hardware wraps; x86 does too, so the results were right, but
the compiler may assume they do not happen. Fixed with unsigned arithmetic, the same bits: `v_mad_i32_i24` (the product of two 24-bit signed factors and
the sum, 2 places), the integer multiply-accumulate of the matrix instructions (`mfma_dot_int`, the tile path's `int64` sum in `matrix_multiply` and the
sparse one, through `wrap_madd` in `amd/src/mfma_math.hpp`), and the 32-bit sum of `mfma_tile_int_scalar`. Test: `test_amd_mfma_math`
`integer_dot_products_wrap_without_overflowing_a_signed_type` (fails under UBSan with the old arithmetic). After the fix the ASan+UBSan run of the whole AMD set
(`ctest -R '^test_amd'`, 162 tests, 2 skipped by design) is green: 1762 s (`test_amd_gcn_tier_same_bits` 1762 s, `test_amd_matrix_fast` 1155 s,
`test_amd_gcn_same_bits` 999 s, `test_amd_gcn_atomics_gfx1250` 677 s, `test_amd_gcn_memory_same_bits` 666 s, the rest under 100 s) at `-j8` on 8 hyper-threads while
another session used the other 8.

**2. The NaN a float instruction leaves was the compiler's choice.** `x + y`, `x * y`, `std::fma(x, y, z)` and `std::fmin/fmax(x, y)` with two NaN operands give the
payload and sign of one of them (the hardware and SSE take the first operand's), and C++ does not say which operand is first. GCC swaps them
as it likes: differently at -O0, -O1 and -O3, differently in two copies of one loop, and `fminf`/`fmaxf` it passes to libm as `(y, x)` even at -O0 (checked in the
disassembly). The recorded digests of `test_amd_gcn_same_bits`, `_memory_same_bits` and `_packed_same_bits` held the choices of the Release build, so those tests were
right in the Release build and wrong in every other one: the same tests fail at plain `-O0` and plain `-O1` without any sanitizer (measured at the head `224874ea`: 50 and 22 instructions of
16806), under ASan 50 (after the overflow fixes), under TSan 82 in CI's run (that one was before `224874ea` fixed the f16 add). It is a real defect of the
interpreter (the result of an instruction depended on the optimization level) and not of the tests. Fixed at the cause: `amd/src/nan_order.hpp` has `nan_add`, `nan_sub`, `nan_mul`, `nan_fma`,
`nan_fmin`, `nan_fmax` with the operand order as a template argument; the result is the plain operation unless it is a NaN with a NaN operand, and
then the first NaN operand of that order, quieted. The orders are the ones that reproduce the recorded digests (found by trying them against the digests in the Release
build with the order of each site chosen at run time, then frozen as template arguments), so no golden digest moved; they are not the same at every site:
`v_add_f32`, `v_mul_f32`, `v_add_f64`, `v_mul_f64`, `v_pk_add_f32`, `v_pk_mul_f32` take the second operand first (GCC commuted them), `v_sub_f32`, `v_subrev_f32`, `v_mul_legacy_f32`
the first; `fma` takes (b, a, c) in `v_fma_f32`, `v_fmac_f32`, `v_fmaak/fmamk_f32`, `v_fma_mix*`, `v_fma_f64`, `v_fmac_f16`, and (a, b, c) in
`v_fma_f16`, `v_pk_fma_f32`, `v_pk_fmac_f16`; `v_max3_f32` and the max in `v_med3_f32` take the library's answer for `(y, x)`; the packed adds of the atomics are `y`-first in both halves for float16 and `x`-first in the low half and `y`-first in the high half for bfloat16, `ds_pk_add_f16` is `x`-first in the low half and
`y`-first in the high half (all found by trying). `nan_fmin`/`nan_fmax` are glibc 2.39's fmin and fmax written out (a tie of the two zeros gives the
second argument, two NaNs the first, quieted, a signaling NaN beside a number that NaN quieted, a quiet one beside a number the number), checked against the library on 9 x 9 values of
both widths. Two more that are not an order: `v_sin_f32`/`v_cos_f32` on gfx942, gfx950 and gfx1250 and `v_fract_f16` pass a NaN through as it came (the float to double to float round trip that would quiet a
signaling NaN is folded away by the optimizer at -O1 and above and not at -O0, and the recorded bits are the folded one's). Test: `test_amd_nan_order` (4 tests: the
helpers are the plain operation for numbers, the order named for NaNs, the library's fmin/fmax where its answer does not depend on the argument order), and the same-bits tests themselves,
which now pass at -O0 (also with ASan+UBSan), -O1, -O3 and with TSan. Not covered by any digest, so the order chosen is a guess that matches the natural one (first operand first): the bfloat16
`ds_pk_add`, `v_fmac_f64`, the scalar float instructions, `v_fmaak/fmamk_f64`, and `v_pk_fma_bf16`; they pass today and are not changed.

**3. Two test defects.** `test_amd_mfma_math` compared `mfma_dot` with a plain `old_sum += a * b` loop; with two NaNs the plain loop's payload is the compiler's, and
`mfma_dot` has the fixed order (that was the TSan failure at -O1 and the ASan one at -O0): the NaN case is now compared with the same sum in the fixed order,
spelled out on the strided layout. `test_amd_gcn_threads` `a_forked_child_launches_on_threads_of_its_own` (and the two `_gfx1100`/`_gfx1030` runs) died under TSan with
"starting new threads after multi-threaded fork is not supported": that is what the test checks on purpose (the pool is rebuilt in the child), so the test binary sets
`die_after_fork=0` through `__tsan_default_options` when built with TSan. Not a finding.

**Merge with `origin/speedup` (e50e35c5).** The other session had meanwhile made the same overflow fixes (`wrap_madd`, kept; `mfma_mac_wrap` is gone), made `test_amd_mfma_math`
compare a NaN as a NaN (mine compares the fixed order exactly, kept), skipped the forked-child case of `test_amd_gcn_threads` under TSan (it runs again, with `die_after_fork=0`), and let the
three digest tests differ in up to 2 % of the shapes in a build without optimization or under a sanitizer. With the order of the NaNs spelled out the digests hold in every build, so that
tolerance is removed again (the three tests are as they were at `224874ea`): an instruction that differs in any build is a failure. The tier test's one seed under TSan is kept.

**Slow, not failing.** Under a sanitizer these tests take long: `test_amd_gcn_tier_same_bits` 1762 s (ASan), 2373 s (TSan), `test_amd_matrix_fast` 1155 s and 1527 s,
`test_amd_gcn_same_bits` 999 s and 1398 s, `test_amd_gcn_memory_same_bits` 666 s and 897 s, `test_amd_gcn_atomics_gfx1250` 677 s and 463 s (loaded rig, `-j8`; CI's own times
were 79 s to fail and 900 s for the tier test under TSan). CMake sets no `TIMEOUT` for them and ctest's default here is none, so none timed out; the CI job's own time limit
is what they eat: the proposal (not made) is a `TIMEOUT` property of 3600 on these five, and moving `test_amd_gcn_tier_same_bits` and `test_amd_matrix_fast` out of the TSan job's list
(they run no threads of their own; `-E` in `ci.yml` line 739) if the job's limit is tight.

## The allocation table and a launch's bookkeeping (branch `sn-memalloc`, from `speedup-next`)

`src/core/memory.cpp`, `include/vgpu/memory.hpp`, `include/vgpu/brlock.hpp`, `src/core/memory_backing.cpp` and the kernarg write of
`run_launch` (`amd/src/hip_api.cpp`). Both vendors' memory accesses and the AMD launch go through them. Every address, quarantine
entry, diagnostic, count and result is as it was: `test_memalloc` holds the answers to those of the code before this change (digest
recorded from it), and the guards of the speed work pass in every build.

### What was measured first

The leads of the previous round held. `tools/bench-amd/profile` (SIGPROF) on a `-g` build of the base: `buf` 27% in
`MemoryManager::scalar_location` (4% in `__tls_get_addr`: the lookup memory is a `thread_local` of a shared library, so every access called
for its address), `MemoryManager::free` 15% and `alloc` 6% of `launch`, `materialize` 14% (a 64 KiB chunk made and zeroed for a 192-byte
kernarg segment), `find_ordinary_miss` and `BigReaderLock::lock` visible. The NVIDIA `launch` workload makes no allocation and spends none
of its time here (the 13% of the earlier note was the AMD launch); NVIDIA's memory workloads reach the table through `find_ordinary`.
A loop of `alloc(192)`, `write`, four loads, a load and a store of a long-lived allocation and `free` (one thread, pinned, minimum of 9)
took 560 ns on the base.

### What changed

* **An allocation does not void what threads remember, and a free voids only what it frees.** Each thread keeps the last eight
  allocations it found, with the table lock's epoch; every `alloc` and `free` took the lock with two epoch changes, so every kernel's
  first accesses after a launch's kernarg `free` went back to the table. Now the lock has a quiet hold (`lock_quiet`, `ExclusiveGuard(l,
  true)`) that leaves the epoch alone, used by `alloc` (a new entry changes no other) and by `free`, which calls `note_free(base)` first.
  The epoch's low 16 bits count frees since its random part was drawn, and a 64-entry log of the bases freed holds the last ones:
  `still_valid(seen, base, &now)` answers whether an entry remembered at `seen` survived the frees since (none of them its own, the
  log not overrun, the random part unchanged), and `find_ordinary_miss` renews such an entry in place of walking the table. Every other
  mutator still takes the full lock. A nested full hold inside a quiet one voids at once and again at its end (`bumped_`).
* **The writer looks at the slots in use.** The lock keeps a mask of the slots some thread has counted in (set the first time, before
  the count, so the Dekker argument of the flag and the counts holds), and a writer waits for those and not for all 64 cache lines.
* **The lock's thread identity and the lookup memory are found from the thread pointer, not through a thread-local.** (The first
  version kept them in initial-exec thread-locals; the core is linked into every shim library, so each library took static TLS, and
  glibc's roughly 1.6 KB surplus ran out in a process that loads several of them: `dlopen` of libamdhip64 or the profiler failed with
  'cannot allocate memory in static TLS block'. `lint_static_tls` now holds every shim to no static TLS.) The lock reads
  `__builtin_thread_pointer()` for identity and hashes it into a slot; the memory of lookups sits in a 1024-slot table keyed by the thread
  pointer (8 probes, claimed by compare-and-swap on a thread's first lookup, given back by a destructor when it ends); a thread that finds
  no free slot, or is ending, reads an empty memory that is never written. The price against the initial-exec version is one dependent
  load per access: about 3-6% on `buf`, `narrow` and `atomics`; the gains of the allocation work on `launch` are untouched.
* **`load_scalar` / `store_scalar` try one inline lookup first** (`hot_location`): the thread's last allocation, no host mapping in
  reach, a chunk that is real. Anything else (another allocation, an untouched or uniform chunk, a fault armed, a miss) takes the
  general path, which is unchanged. `run_allocation` (`load_run`, `store_run`, which the AMD engine uses for contiguous lanes) uses the
  lookup memory too, where it took the lock and walked the map on every call.
* **Nodes and chunks come back.** `free` keeps the node of `live_` (with the allocation's array of chunk slots) for the next `alloc`,
  `freed_` keeps the node of the entry the quarantine evicted for the next free, the eviction order is a deque of iterators (it was a
  map popped only at its front), and inserts take the end as their hint (addresses are a bump). A chunk of an allocation that was
  never written past its end goes back to `backing` with `release(chunk, dirty_bytes)`: below 4 KiB dirty it is cleared there and kept
  (at most 64) and the next `acquire()` hands it out, counted as in use again, without a 64 KiB zero. Every write is bounds-checked
  against the allocation's size, and a chunk made whole from a uniform chunk is a whole chunk of the allocation, so for the last
  chunk the bytes past `size` are still zero.
* **The kernarg segment is written, not copied through a padded vector.** A new allocation reads zero, so `run_launch` writes the
  arguments only.

### Results

Alternating runs, 11 per cell, one host thread pinned to CPUs 12-13 (`--threads all`: CPUs 8-15), `flock /tmp/speedup-bench.lock`,
load 3 to 5 on server2 (32 threads, shared). Ratio is B/A of the medians (base `90bf40da`, this branch), p is Mann-Whitney, counters
identical in every pair (the NVIDIA `atomics` counters at 8 threads differ between two runs of the base too: a spin loop).

| workload | MI300X, 1 thread | RX 6800, 1 thread | MI300X, all | RX 6800, all |
| --- | ---: | ---: | ---: | ---: |
| `launch` | 1.319x (p 0.000) | 1.116x (0.001) | 1.303x (0.000) | 1.212x (0.000) |
| `launch_sync` | 1.474x | 1.151x | | |
| `graph` | 1.228x | 1.167x | | |
| `narrow` | 1.246x | 1.190x | | |
| `atomics` | 1.167x | 1.145x | | |
| `buf` | 1.139x | 1.134x | 1.148x | 1.120x |
| `mem` | 1.056x | 1.036x | 1.059x | 1.045x |
| `stream` | 1.028x | 1.033x | 1.055x | 1.030x |
| `api` | 1.025x | 1.029x (same) | | |
| `flat` | 1.019x (same) | 1.019x (same) | 1.026x | 1.025x |
| `smem`, `ldsx`, `launch_mid`, `launch_wide`, `apicheap`, `memcpy` | same (0.98x to 1.01x) | same | same | same |

NVIDIA (RTX 3060 profile, SASS and PTX, 11 runs): `atomics` 1.053x SASS and 1.063x PTX (p 0.000 and 0.002), `mem` 1.018x / 1.019x at
one thread (p 0.000, under the 2% bar: "same") and 1.028x at 8 threads (faster), `texture`, `launch`, `shared` same. `local` SASS reads
0.975x (p 0.000) at one thread and 0.968x at 8: it does not use the memory manager (`h_local` works in the warp's own vector), and
the same loss appears in builds of the base to which only dead functions were added in front of `load_scalar` (90 and 160 lines of
volatile arithmetic in `memory.cpp`: 0.975x each), and not with 12 or 40 lines, with 24, 200 or 1000 bytes allocated at start-up, or in
the commit before `hot_location` (all 1.00x). That is a code-placement effect (not located further), so it is not dropped for a cause
in this change. `local` on PTX: same.

`alloc`+`write`+loads+`free` of the loop above: 560 ns to 179 ns (3.1x, min of 9, one thread); a load and a store in one allocation
8.1 ns to 5.5 ns.

### Tests (wired into CMake)

`test_memalloc` (13 tests; also run under ASan+UBSan and TSan): a model of the bump allocator with a bounded quarantine
(addresses, `OutOfMemory`, `DoubleFree`, `InvalidFree`, `InvalidPointer`, `UseAfterFree` and `OutOfBounds` for 33,000 random operations
on four configurations, `used()` and `live_allocations()` after every one); the quarantine's exact bound (4096 entries: the 4097th
oldest is "retired", the rest are double frees); the digest of 12,000 to 40,000 random operations (values, addresses, error codes and
texts, `used()`, `resident_bytes()`; four configurations including OOM and the quarantine's bound) equal to the one the code before this
change gives (recorded from `90bf40da` with `-DVGPU_MEMALLOC_DIGESTS_ONLY`); a freed allocation read through the thread's memory after
0 to 140,000 other frees (the count wraps at 65,536) is a use-after-free and its neighbours stay right; six remembered allocations
freed in random orders; four readers beside two threads that allocate and free; a new allocation reads zero, in its slack too,
after the last one held scalar writes, a copy, a uniform fill and a write over a uniform chunk (3 page sizes, sizes from 1 byte to 200,000);
the backing's counts and zeroing for chunks given back nearly clean; `resident_bytes()` through 300 allocate-write-free cycles; the lock:
a quiet hold leaves the epoch, a full one changes it (nested both ways), `note_free`/`still_valid` (other bases survive, the freed one and
more than 64 frees and a full hold and the count's wrap do not), and exclusion of 80 reader threads by quiet and full writers.

Mutations (scratch copies, each run on a clean rebuild), what fails: no `note_free` in `free`: the digest, the stale-read test, the
frees test; `still_valid` ignoring the log: the same and the epoch test; recycled node keeping its chunk pointers: aborts (double
free); pool clearing one byte short or not at all, the last chunk's dirty bound one byte short: the digest, the zero test, the pool test;
quarantine bound off by one: the quarantine test; the writer ignoring the slots in use or readers not announcing their slot:
segmentation faults in the first test with readers beside a writer (the exclusion tests would fail); a full hold nested in a quiet one
not voiding: the epoch test. Not killed, by construction: the wrap handling of the count (an unhandled carry only changes the random part,
which `still_valid` treats as a full void) and not reusing the quarantine's node (speed only); the `seq` comparison on eviction guards a
pointer freed twice, which the bump allocator never produces.

Also passed on this tree: `test_memory`, `test_memfast`, `test_memrmw`, `test_memrmw_lockfree`, `test_worker_pool`, the guards
(`test_amd_gcn_same_bits`, `_memory_`, `_tier_`, `_packed_`, `test_amd_codeobject_same_bits`, `test_amd_matrix_fast`, `test_amd_mfma_math`,
`test_amd_nan_order`, `test_amd_packed_math`, `test_amd_gcn_threads`, `test_amd_hip_queue`) and `e2e_sass_tier` in Release, ASan+UBSan (-O0) and
TSan (-O1) (the sanitizer runs build without `VGPU_WERROR`: GCC 13 rejects `atomic_thread_fence` under `-fsanitize=thread -Werror`), every
`test_*` and the AMD end-to-end set and `e2e_sass_*` in Release (267 of 267), `bench-amd --check-counters` (36 pairs match). The NVIDIA
`bench --check-counters` skips on this host (its nvcc builds different device code than the recorded counters); `--ab` reports identical counters.

### Not done, and negative results

* **`local` on SASS reads 2.5% slower** in these builds with nothing of it in this change (above): a code-placement effect.
* **`load_scalar` of an untouched or uniform chunk, a miss and the fault-armed case** take the general path; the hot path is the real
  chunk in the thread's last allocation, which is what the lanes of one instruction do.
* **A flat sorted array for the quarantine** (`freed_`): inserts at the end and evictions at the front would be free, but the access
  pattern is the program's and a free in the middle moves up to 64 KiB; the map with recycled nodes keeps the worst case.
* **Sizing the chunk to the allocation** (a 192-byte allocation holding 192 bytes): `kChunkSize` is assumed by every read and by
  `resident_bytes()`, `ram_bytes()` and the spill file; recycling the 64 KiB chunk keeps them.
* **A dirty-extent flag for chunks made from uniform ones** (`wide_dirty`) was written and dropped: a uniform chunk exists only as a
  whole chunk of an allocation (`fill`), and the last chunk of an allocation that is a whole chunk has its whole size as dirty bound, so
  no input reaches it (the mutation that removes it cannot fail a test).
* **`SIGPROF_STACK=1` on the AMD `launch` workload** hung twice in five runs of one `-g` build (all three threads in a futex wait) and
  never in 350 runs without the sampler (150 `launch`, 100 `launch_mid` and 100 `launch_wide` at 8 threads) or in 16 more with it:
  the sampler's unwinder takes the loader's lock inside a signal handler. Not a finding of the memory manager; noted so that the
  profile is read with that in mind.
## AMD speed gaps closed (branch `sn-amdgaps`)

**Matrix tile path.** The f8f6f4 forms (`v_mfma_f32_16x16x128_f8f6f4`, `v_mfma_f32_32x32x64_f8f6f4`) and their block-scaled twins, BLGP's lane permutation
of B, the doubles' BLGP negations and the four-block `v_mfma_f64_4x4x4_4b_f64` now take the tile path (`matrix_multiply_fast`, `matrix_small_k`, scale
tables by block of 32 K and row). `mfma_scale_mul` (mfma_math.hpp) fixes the order of the two operands of the scale multiply, used by both paths. A bug of
the reference path found on the way: its binary16 shortcut ignored BLGP's permutation of B. New bench-amd workloads `mfma_f8f6f4`, `mfma_scale` (target
`mi350x`, gfx950) and `mfma_f64_4b`. A/B (9 runs, one thread, CPUs 24-31, server2 load 3 to 25): mfma_f8f6f4 7.7x, mfma_scale 8.0x, mfma_f64_4b 2.7x, p < 0.001,
counters identical. 8-bit float operands decode each register word once (mfma_fp8 1.15x of the base).

**Tier coverage** (`VGPU_AMD_TIER_STATS=1`). Over the 46 pantheon workloads (run-pantheon-workloads.sh) the interpreter ran 7.1% of the instructions (574 M of
8.07 G: `v_xor_b32_sdwa`, `s_load_*`, `v_cndmask_b32_e64`, `s_mul_hi_u32`, `s_endpgm`, `v_pk_fma_f16`, 29 M NaN handbacks of `v_fma_f32`); rocBLAS's test
program 12.7%. Added: SDWA integer forms, VOP3 forms of existing handlers, `v_cndmask_b32_e64`, scalar ALU leftovers (`s_mul_hi`, `s_brev`, `s_cmpk_*`,
`s_bitcmp*`, `s_getpc`), `s_load_*`/`s_endpgm` through the Machine without step()'s frame, one-input float functions (single and double), 16-bit integer and
half forms on the CDNA parts, class and 16-bit compares, `v_pk_fma/add/mul/min/max_f16` (routed to packed_math.hpp), `v_pk_mov_b32`, `v_div_scale/fmas/fixup`,
`v_fma_mix*`, `v_accvgpr_read/write`. After: 0.5% of the pantheon instructions (MFMA, atomics, a few 16-bit forms), 7.1% of rocBLAS's. A/B on bench-amd:
smem 1.14x, valu_int 1.03x, atomics 1.04x, others unchanged.

**NaN-aware float path.** A NaN result of `v_add/sub/mul/fma_f32`, the double forms and `v_pk_*_f32` is worked out again in the tier by the interpreter's rule
(`nan_add<1>`, `nan_sub<0>`, `nan_mul<1>`, `nan_fma<2>`, `nan_fma<0>` for the packed fma); one-input functions of the host's libm pass their NaN through. The 29 M
handbacks came from the pantheon virus workloads (random data). Min/max, nearbyint, angles in turns and `v_div_fmas` still hand a NaN back. The tier test gained
flavour 17 (halves with specials and signaling NaNs) and generated corpora `isa_corpus_tier_<gfx942|gfx90a|gfx950>.txt` (amd/tools/tier-corpus.py; tier on against
tier off, no recorded digests); they caught bugs of this branch's own (`v_fract_f32` NaN, clamp on cvt_f32_* off the guide parts, a negated signaling half NaN in class,
`v_div_fmas_f64` under -O1).

**Packed min/max.** `v_pk_min/max_f16` run whole-wave (`packed_minmax_lanes`; `pk_fminf`/`pk_fmaxf` and the guide parts' bit rule now live in packed_math.hpp;
unit test against the per-lane code). The dot2 NaN lanes are unchanged.

### Not done, and negative results (this branch)

* Trampolines for the vector memory instructions the tier has no handler for (buffer, flat, scratch, LDS, atomics) through the Machine's functions without
  step()'s frame: no measurable gain on mem, flat, buf, ldsx (p > 0.05); dropped.
* NaN lanes of `v_min/max_f32` (MODE.IEEE decides) and the dot2 NaN lanes stay with the interpreter.

