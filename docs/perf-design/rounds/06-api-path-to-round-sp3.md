## The API calls' own cost (branch `sp-apipath`, from `d8ec6128`)

What a program pays for a runtime call that is not simulation: `cudaMalloc`, `cudaMemcpy`, `cudaEventRecord`, `hipMemsetAsync`, `hipFree`.
The harness rows are `api` (NVIDIA: 15 calls a round; AMD: 13 calls a round, 5 of them hand-offs to a stream's thread), `apicheap`,
`memcpy`, `stream`, `launch_sync`, `graph`. The numbers below come from the pinned benchmark CPU (`--cpus 6`), so on the AMD side the
stream threads share that one CPU and every hand-off is two context switches (`VGPU_QUEUE_SPIN` is 0 there); wall-clock gains on the AMD rows
are therefore much larger than the cycle counts, which are given beside them.

### Where a call's time went (perf, one host thread, 4 KiB buffers)

* NVIDIA `api`: 19% was the zeroing of a fresh 64 KiB chunk (`backing::acquire`, libc `rep stos`) for each `cudaMalloc` of 4 to 6 KiB: the
  pool of chunks given back nearly clean took back only 4 KiB, and these buffers dirty more. Then `ApiCall::ApiCall` 5%, `__tls_get_addr` 3 to 5%
  (nine thread-locals of the profiling core, each its own lookup), the clock 4%, `pthread_mutex_*` 5%, the telemetry writers 4%, and a per-call copy
  of the arguments into a `KeptArgs` nobody reads. A call is `traced_call`'s `ApiCall` and `guard`'s, nested.
* AMD `api`: 70 mutex acquisitions a round (13 calls), ~15 futex calls, 6.4 context switches. `hipFree` alone took 14 locks (every queue
  asked `tail()`, `wait()` and `take_error()` under its mutex, though idle).

### What changed

* **Profiling core** (`include/vgpu/profiling.hpp`, `src/core/profiling.cpp`). The thread-locals are one struct (`ThreadApi`), constant-initialized
  (otherwise every access asks whether the thread's `__tls_init` has run) and found once per call (`ApiCall` keeps the pointer: its destructor
  and `silenced()` do not look it up again). `ApiCall`'s work for a call somebody listens to, or that has arguments noted, is out of line
  (`begin`, `end`); for the others the constructor is a dozen instructions. `ApiCall`'s constructor reads the two flags directly (they are the file's own; `enabled()` and `hooked()` stay exported
  functions: a flag in the header, inline or hidden, is one object for each library, and the CUPTI shim's markers and the driver's records read one nobody set;
  `test_cupti_activity_flags`). `note_context_made()` reads before it exchanges.
* **Runtime and driver** (`nvidia/src/runtime_api.cpp`, `driver_api.cpp`). The arguments of a call are copied into `KeptArgs` only when a recorder
  or subscriber is on (an `optional` in the frame). `capture_unsafe_gate()` (every allocation, copy and launch) reads an atomic that mirrors
  `g_captures` before it takes `g_graph_mu`; the five places that change `g_captures` update it.
* **Telemetry** (`src/core/telemetry.cpp`). The `note_*` calls refresh under the lock they hold (`refresh_locked`), so the writers' mutex is a plain
  one; the segment's sequence number is stepped by a load and a store under that mutex (fence and release store for the ordering) instead of two locked increments.
* **Memory backing** (`src/core/memory_backing.cpp`). The pool of chunks given back keeps those with up to 32 KiB written (`kPoolMaxDirty`, was 4 KiB):
  cleared under the lock to the byte promised, as before. `pooled_chunks()` for the test.
* **HIP queue** (`amd/src/hip_queue.hpp`). `idle()`, `tail()` and `take_error()` answer from atomics without the queue's mutex (`idle` reads the
  completed count before the submitted one; a failure is noted before its item counts as done); `wait()` of a finished item returns without the
  lock. `run_inline` takes any callable and can tell the item's number.
* **HIP runtime** (`amd/src/hip_api.cpp`). `hipEventRecord` on a stream with nothing in it (and every marker it follows done) reads the clock on
  the caller (a profiler attached keeps the stream's thread); the result of a waited-for item stays in `in_order`'s frame (a `shared_ptr` and a second
  function object less); `hipFree` takes the lock once to find the owner and the queues; the three synchronizations take the lock for
  `release_to_thresholds` only once a pool exists (`g_pools_made`). **A copy or fill of at most 256 KiB (`kInlineMemoryBytes`) queued on a stream with
  nothing in it, nothing it follows unfinished and no profiler attached, is done in the call** (`in_order`): `hipMemsetAsync`, `hipMemcpyAsync`
  and the device-to-device copy and fill that HIP queues and returns from at once. A failure is the stream's, kept for the next synchronization, as
  for a queued item. Kernels, host functions, events behind work and larger copies are queued exactly as before. This moves *when* such work happens
  (at the call, never later) and starts no stream thread for a stream used only this way; a program cannot tell except by timing or by a race.

### Results

15 alternating runs, `--cpus 6`, load 12 to 55 on the server (the A/A column is the base against a copy of itself: ratios of medians, all p > 0.1).

| workload | MI300X 1 thread | MI300X all | RX 6800 1 | RX 6800 all | A/A |
| --- | ---: | ---: | ---: | ---: | --- |
| `api` | **14.5x** | **14.5x** | **14.7x** | **14.7x** | 0.96 to 1.06 |
| `memcpy` | **4.96x** | **5.10x** | **5.22x** | **5.13x** | 0.97 to 1.05 |
| `apicheap`, `stream`, `launch_sync`, `graph` | 0.99 to 1.00 | 0.99 to 1.01 | 0.99 to 1.01 | 1.00 to 1.01 | 0.97 to 1.03 |

| NVIDIA (RTX 3060 profile) | SASS | PTX | A/A |
| --- | ---: | ---: | --- |
| `api` | **1.48x** (p < 0.001) | **1.48x** (p < 0.001) | 1.02 / 1.00 |
| `launch` | 1.03x (p 0.14) | 1.03x (p 0.004) | 1.00 |
| `startup` | 1.00x | 1.00x | 1.01 |

Counters are identical in every pair. The same builds under `perf stat` (user instructions, cycles, context switches of the process; 3 runs, spread under 1%):

| | instructions | cycles | context switches |
| --- | ---: | ---: | ---: |
| NVIDIA `api`, base | 1187 M | 615 M | 5 |
| NVIDIA `api`, now | 959 M | 414 M | 5 |
| AMD RX 6800 `api`, base | 2.43 G | 1.6 to 2.0 G | 640 k |
| AMD RX 6800 `api`, now | 1.52 G | 0.55 G | 6 |
| AMD RX 6800 `memcpy`, base | 0.70 G | 1.29 G | 197 k |
| AMD RX 6800 `memcpy`, now | 0.47 G | 0.75 G | 3 |

Where it comes from, as the changes were added (NVIDIA `api`, wall clock, 15 runs against the base): the call-path, capture-gate, telemetry-mutex and chunk
pool changes together 1.34x; the thread-local struct, `ApiCall`'s fast path and the sequence number then took the cycles from 456 M to 414 M (1.48x). The
profile before any of it put 19% of the cycles in the chunk zeroing. AMD `api`: the lock-free queue methods and the inline event record alone left the process
switching 600 k times (wall clock 1.1x); the inline small copies and fills took the switches to 6.

### Tests (wired into CMake)

`test_profiling_api` (the recorder and subscriber contracts of `ApiCall`: outermost call, nesting, correlation, whose arguments are whose, silence, threads,
the context hook once; the flags), `test_runtime_capture_gate` (a global capture refuses an allocation on any thread and the gate opens after it; relaxed
does not refuse), `test_telemetry` (two steps of the sequence number for each update, even between), `test_memalloc` (chunks with up to 32 KiB dirty are
kept, cleared and counted; the limit is pinned by a `static_assert`), `test_amd_hip_queue` (`idle`/`tail`/`take_error`, a failure seen the moment its item is
seen done, `run_inline` with a lambda and the item's number), `test_amd_hip_event_inline` (a record on an idle stream is complete at the call, also with the
host refusing threads; behind a host function, and on a blocking stream behind the null stream's work, it is not), `test_amd_hip_async_inline` (a small fill
and copy on an idle stream need no thread and are done; behind a running host function they are not; a megabyte is queued), `test_amd_hip_pool_sync` (each
of the three synchronizations gives a pool's excess back, none made before), `test_amd_hip_thread_refused` (now with a host function as the first item).

Mutations (a scratch copy, each rebuilt and run; what fails): every call outermost; noted argument count not cleared; the fast path ignoring noted
arguments; `set_enabled` writing the other flag: `test_profiling_api`. Capture flag never set: `test_runtime_capture_gate`. `begin_update` storing nothing:
`test_telemetry`; `note_memory` calling the locking `refresh` under the now plain mutex: `test_telemetry` (deadlock, killed by timeout). Pool limit 4 KiB:
`test_memalloc` (the `static_assert`); the pool clearing one byte short: `test_memalloc`. `take_error` always ok, `idle()` always true (timeout),
`run_inline` not telling the number, a failure noted after its item counts as done: `test_amd_hip_queue`. Event record never inline; ignoring what the stream
follows: `test_amd_hip_event_inline`. Async copies never inline; no size limit; everything inline, host functions too: `test_amd_hip_async_inline`
(and `test_amd_hip_thread_refused` for the last). Synchronizations never giving a pool back; the flag never set: `test_amd_hip_pool_sync`.

### Not done, and negative results

* **TLS descriptors** (`-mtls-dialect=gnu2`, which replaces the `__tls_get_addr` call with a descriptor call): `api` 1.034x SASS and 1.055x PTX
  (p 0.003 and < 0.001) on the tree before the thread-local work above, `launch` same. Not kept: it needs the loader of the machine that runs the shims to support
  TLS descriptors (glibc's x86-64 support is not something to assume of every distribution the shims are installed on), for a few percent.
* **One `ApiCall` for a call that `traced_call` and `guard` both open**: the inner one is a nested no-op, but dropping it needs every `guard` caller
  audited for a public entry that opens none of its own.
* **A waiter running the queued work itself** (instead of sleeping while the stream's thread does): the wake-up at submission (a futex call and, on one CPU,
  a switch to the woken thread) stays, so the gain would be the second switch only, and the work would run on another thread than the stream's.
* **Stream threads kept for reuse**: `stream` creates and destroys a stream a round, so starts a thread a round (8000 `clone3` calls a run); a parked
  thread would carry thread-locals from one stream to the next (the current device, the last error).
* **`hipEventSynchronize` and `hipEventQuery`** each take three event/runtime locks to look at one event (`in_capture`, `remote_event`, `event_marker`); one
  look under both would save two lock pairs of about 40 a round: not done.
* The AMD `launch_sync`, `stream` and `graph` rows are the kernel launch (`set_up_group` is 55% of `launch_sync`), not the call.

## The AMD engine: whole-wave memory accesses (branch `sp-amdmem`, the `speedup` round)

The memory instructions of the AMD engine went to the memory manager a lane at a time: every lane of a gather, a buffer access or an atomic looked up its allocation
(`lookups()` and the table behind it), asked `HostMaps::may_contain`, and made its own `load_scalar` / `store_scalar` (an atomic made two, one for the load and one
for the store). The profile of `buf` (RX 6800) was `load_scalar` 25%, `store_scalar` 18%, `buffer_access` 20%, `load_words` 7%, `may_contain` 4%; of `atomics`,
`global_access` 20%, `load_scalar` 18%, `atomic_rmw` 6%, `store_scalar` 4%, and with a single lookup for the load and the store `Machine::atomic_rmw` alone was 47% of the
MI300X run, nearly all of it the chain of dependent loads that finds the allocation (`lookups()`, the thread's last way, the chunk table). Now a wave's accesses go to the memory manager together wherever the manager can take them as the calls
they stand for, and decline (having done nothing) wherever it cannot, so the lanes then go one at a time and meet any error where they would.

Measured on server3 (Ryzen 7 PRO 8840U, one SMT sibling of one core, the machine at a 1-minute load of 30 to 130 from other builds), Release, GCC 13.3, simulated
MI300X and RX 6800, one host thread, `tools/bench-amd` A/B against `d8ec6128`, 11 alternating runs; ratios are B/A of the median (the best-of ratio agrees within
3% in every row). Every row's counters are identical, and `bench-amd --check-counters` matches (37 pairs). The A/A control of the same harness (base against the
same base, 9 runs) gave medians between 0.964 and 1.044 (`memcpy` mi300x 0.964, `ldsx` rx6800 1.044, the rest within 2%).

| workload | what it runs | mi300x | rx6800 |
| --- | --- | ---: | ---: |
| `buf` | buffer dword loads and stores, some past the end | **2.74x** | **2.37x** |
| `narrow` | byte and short global loads and stores | **1.78x** | **1.70x** |
| `mem` | 16-byte copy, scalar scale, gather | **1.58x** | **1.36x** |
| `lds` | tree reduction and tile transpose (global rows, LDS words) | **1.45x** | **1.27x** |
| `flat` | generic pointers that reach global memory | **1.33x** | **1.16x** |
| `atomics` | global atomic add, max, CAS loop | **1.23x** | **1.24x** |
| `ldsx` | wide LDS accesses and LDS atomics | 1.13x | 1.05x (p 0.065, same) |
| `smem`, `memcpy`, `stream` | not changed | 1.00, 1.00, 1.00 | 0.99, 0.99, 1.00 |
| `valu_int`, `launch` (controls) | not changed | 0.99, 1.01 | 0.99, 0.99 |

p < 0.001 for every "faster" row. Eight host threads were not measured (the benchmark is pinned to one CPU on this host); the threaded atomics path is held by the
tests below, in Release and under ThreadSanitizer.

NVIDIA: the SASS and PTX engines share `batch_ok` / `load_batch` / `store_batch`, which got the one-pass test below. `tools/bench` A/B, 9 runs, counters
identical in every row: `mem` ptx 1.04x (p 0.004), the rest of mem, shared, atomics, texture, local, int_alu, divergent and warp within 2.5% of the base,
`atomics` ptx 0.97x on a 85 ms workload whose spread is 11 to 17% (p 0.006, best-of 0.985; not beyond the harness's 2% on both).

### What changed

* **`MemoryManager::batch_ok`** (`src/core/memory.cpp`) makes one pass over the addresses for the three tests every access must pass (alignment by the OR of the
  addresses, inside the first access's allocation by the largest offset, and no host mapping in reach by `HostMaps::may_overlap(lowest, highest)`, which rules a
  range wholly inside the device windows out as one address, and otherwise asks `may_contain` of each as before). `load_batch` takes the width as a constant.
* **A wave's gather or scatter** (`Machine::wave_scalars`, called from `tier_global` in `gcn_tier_machine.inc` and from `global_access`): the active lanes that
  make no run (or the lanes after the run that ended) are one `load_batch` / `store_batch` of bytes, shorts, words or pairs; lanes of a load after a lane with the
  same address take its value (the load memo's rule).
* **Buffer dword loads and stores** (`Machine::buffer_words`): the wave's addresses and whether each is inside the resource, then the lanes inside as one batch; the
  lanes outside read zero or write nothing, as before. `fits` moved out of the lane loop (it reads `index`, now declared before it).
* **Contiguous runs** (`Machine::memory_run`): the run is found from the address registers' rows (the lane limit from EXEC at once, a 64-bit compare per lane),
  and a run of one word a lane in a vector register is moved straight to or from the register's row (`load_run` into the row, `store_run` from it); two and four
  words a lane are transposed row by row.
* **Atomics** (`MemoryManager::rmw_batch`, `Machine::wave_atomics`): a wave's atomics are one call, lane by lane in order, with one lookup for the lot: the plain
  load and store-if-changed on one thread, the compare-and-swap of `atomic_rmw` (one `atomic_rmw_slot` per lane, the loop that was `atomic_rmw_fn`'s) across
  threads. The old values go to the destination register after the call, or, for the lanes made before a store that could not get host memory, before the error
  leaves. A single atomic (`Machine::atomic_rmw`, the fallback and the scalar and buffer forms) uses `MemoryManager::rmw_if_changed` on one thread: the load, the
  operation and the store of a changed result at the word `hot_location` finds, without a second lookup. The data and the old value are read and written in the
  register rows.
* **`h_ds`** (the tier's LDS handler): the bounds test of the lanes is branch-free and `ds_read_b32` / `ds_write_b32` have loops of their own (the other widths
  keep the general one).

### Tests (all wired in CMake; each shown to fail under the mutation next to it, built and run on server3)

| test | what it holds | mutations that fail it |
| --- | --- | --- |
| `test_amd_gcn_memory_same_bits`, new `the_wave_wide_memory_paths_give_the_bits_lane_by_lane_gives` | every memory shape of the eight ISA corpora (DS, global, flat, buffer, scalar) in the twelve address modes, with half the lanes off in some: digest with the whole-wave paths on against off (`amd::set_fast_paths`), with no recording to trust | batch stores in reverse lane order (181 of 16068 instructions differ); narrow batch loads not sign-extended (114); buffer lanes outside the resource keep their register (28); atomics do not give the old value back (77); each also fails the recorded-digests test |
| `test_amd_gcn_memruns` (existing) | runs of lanes against lane by lane, 18 address patterns and 7 masks, 7 kernels, 5 targets | the run reaches across lanes that are off (`copy32s`, mask 1); a load run one lane late (every pattern) |
| `test_memrmw`, new `a_wave_of_atomics_as_a_batch_matches_the_same_accesses_one_at_a_time`, `..._declines_with_faults_armed_or_stuck_cells`, `a_single_thread_atomic_stores_only_a_changed_word_and_only_in_a_real_chunk`, and `changed_new`/`changed_old` in the random run | `rmw_batch` (plain and lock-free) and `rmw_if_changed` against the calls they stand for: 24 000 batches of 1 to 64 accesses of 1, 2, 4 and 8 bytes (repeats, chunk and allocation edges, misaligned, other allocations, host memory, uniform, real and untouched chunks, an operation that throws), values, errors and the accesses made before them, memory, chunks made real after every round; declines with faults armed | the plain batch stores an unchanged result (chunks made real differ, round 7); the lock-free batch stores always (round 0); `rmw_if_changed` ignores an armed fault |
| `test_memfast`, new `a_batch_declines_where_a_host_mapping_covers_one_of_its_addresses`, mode 10 of the batch fuzz (a misaligned lane inside the allocation) | `batch_ok`'s three tests | host-map test removed; alignment test removed (`MisalignedAccess` out of the reference call); allocation-bound test removed (segmentation fault) |
| `test_amd_gcn_tier_same_bits` (existing, `VGPU_TIER_ONLY=ds_write`) | the tier's handlers against the interpreter | `ds_write_b32` with the second offset (21 of 1140 runs differ) |

Also run on this tree, Release: `test_amd_gcn_same_bits` (and the `simd_base` twin), `_packed_`, `_tier_`, `_memory_`, `test_amd_codeobject_same_bits`,
`test_amd_matrix_fast`, `test_amd_mfma_math`, `test_amd_nan_order`, `test_amd_gcn_memruns` (five targets), `test_amd_gcn_atomics` (four), `test_memfast`, `test_memory`,
`test_memrmw`, `test_memrmw_lockfree`, `test_atom_rmw`, `test_memalloc`, `test_memcount_forms`, `test_sass_access_count`, and the AMD execution tests (89 of 89:
`test_amd_gcn_{exec,memory,bytes,int64,mixed,half,calls,builtins,packed,spill,crosslane,doubles,divide,narrow,lds,idioms,asm,cdna_gaps,abandon,threads,counters,mem_facts,globals,grid,ops}`
on every target they have, `test_amd_hostcall`, `test_amd_hip_queue`). Under ASan+UBSan (-O0, `halt_on_error`, leak detection on): `test_memrmw`, `test_memfast`,
`test_amd_gcn_memruns`, `test_amd_gcn_atomics`, `test_amd_gcn_memory_same_bits` (both tests), and `test_amd_gcn_tier_same_bits` restricted to `ds_`, `global_`,
`buffer_` and `flat_` (the full tier test at -O0 was not run). Under TSan (-O1, `setarch -R`): `test_memrmw`, `test_memrmw_lockfree`, `test_amd_gcn_memruns`,
`test_amd_gcn_atomics` (the threaded atomics). `e2e_sass_tier` and `e2e_sass_archs` skip on server3 (no nvcc): not run.

### Not done, and negative results

* **Wide scalar loads as one `load_run`** (`s_load_dwordx2` .. `x16`): `smem` 1.015x (p 0.30) on the MI300X and 1.038x (p 0.11) on the RX 6800; that workload is the
  scalar ALU's, `load_scalar` is 1.4% of it. Dropped.
* **`OpName::operator==` with the hash test inline and the text test out of line** (so the compare is small enough to be inlined at its uses): `lds` rx6800 1.038x and `flat`
  mi300x 1.041x (p < 0.001), the other fourteen rows of `lds`, `ldsx`, `mem`, `flat`, `valu_int`, `divergent`, `wave`, `buf` the same. Dropped: a gain under 4% in two rows,
  and a change of no behaviour, which no test can fail.
* **A single atomic through `rmw_scalar`'s `rmw_locate`** (the first version of the one-lookup atomic): `atomics` mi300x 0.948x (p 0.05), rx6800 1.004x, slower than the two
  calls: `rmw_locate` is out of line and checks the faults, the maps and the allocation on its own. `rmw_if_changed` on `hot_location` measured 1.058x / 1.046x before the
  batch (and the batch is what is kept for a wave).
* **`HostMaps::may_overlap`** is conservative where the maps' range and the addresses' range overlap (each address is then asked, as before): a program with a pinned
  buffer whose range spans the device windows (`in_device_window` set) gets no gain from it.
* **Not batched**: buffer atomics, byte and short buffer accesses, formatted (MTBUF) accesses, dwordx3 and dwordx4 gathers (a dwordx4 is one 16-byte access each, which
  `load_batch` is not), anything with peers attached, a load after a run that read the lane's address (the load memo), and every access that reaches LDS or private memory
  through a flat address.
* **Bench noise**: the pinned CPU's sibling was idle, but the host's load was 30 to 130 throughout; a run of 9 to 11 alternating pairs held the medians within 2% (A/A above).

## The SASS engine outside the JIT: diverged groups, barriers, atomics and memory (branch `sp-sassctl`, base d8ec6128)

Profiled first (one host thread, `cycles:u`, the RTX 3060 profile, `perf annotate` on a `-g` build): `divergent` spent 14% in `tier_lanes_at` and
only 35% of its runs were chained; `shared` 12% in `tier_release_bar` (a lane-by-lane loop over the block's warps) and a `switch` on the load's
sub-word extension in every lane of `h_shared`/`h_global`; `atomics` left the tier for `execute()` and spent 26% of `exec_mem` on a 64-bit
**division** (`addr % n`, n = 4 or 8) per lane, an O(n^2) search for the distinct addresses, a `host_address` call per distinct address, and up to 26
mutex pairs per instruction (the stripe was one per 8 bytes); `mem` called `MemoryManager::resolve` once per lane of every 16-byte access;
`warp` did 32 scalar lookups per shuffle. The ALU handlers and the SIMD kernels were not touched.

### What changed

* **The chain goes on into a diverged warp** (`tier_run_t`, tail): when the runnable lanes stand at several addresses and nothing asks the scheduler
  for more (no slice, give-way, WARPSYNC set or forced mask), the loop picks the group at the lowest address, as `step_warp` does, and goes on into it
  if it is in this section. `divergent`: 99.98% of the runs are chained (was 35%). `tier_lanes_at` (now `exec::sasslanes::lanes_at`,
  `include/vgpu/exec/sass_lanes.hpp`) compares four lanes per movemask.
* **Atomics** (`exec_mem`, `include/vgpu/exec/sass_atomics.hpp`): the lanes' distinct addresses come from a 64-entry hash instead of a scan; the
  alignment test is a mask; the host-mapping probe is gone (`load_batch` already declines every address `host_address` would answer); the stripe locks are
  held by a `StripeLocks` (no `unique_lock` array) and are per cache line (`atomic_stripe`: a warp's 32 consecutive words take two locks, not sixteen);
  operands are read as rows; the lane loop of each common operation (add, f32 add, min and max of unsigned, and, or, xor, exchange, compare-and-swap)
  is its own loop over a named functor (`sassatom::*Step`, chosen by `step_for`), the rest runs `sassatom::apply`, the former lambda. ATOMG, ATOM, ATOMS and RED
  are a tier form (`Form::Atom`) whose handler is `exec_mem`: the run loop counts and advances them, `execute()`'s probe of `fast_alu` and its dispatch are gone.
* **Shuffles** (`h_shfl`, `tier::shfl_shape`, `tier::shfl_move`): a full warp whose lane and clamp are the same in all lanes, with a segment mask of 0 and a
  lane under 32 (every `__shfl_xor/down/up/_sync` of a loop's offset), moves its words as a butterfly (`pshufd` and vector swaps), a shift or a splat; the
  in-range predicate is worked out for the warp. The analysis does it for constant lanes, the handler for a lane held in a register (`row_uniform`).
* **Barriers** (`sasslanes::wait/waiting_on/wake`): arrival and release of BAR and BSYNC move the 32 kinds, arguments and addresses a vector at a time;
  a block keeps its live thread count (`Block::live`, set by `init_block`, lowered by both EXITs) instead of counting the warps' lanes at every arrival.
* **Memory handlers**: `h_shared` moves a full warp's 32 consecutive words (or one word every lane names) as a block and counts them as one pass (`count_shared`,
  also the tail of `count_access`); `h_global` moves a coalesced run of words straight between memory and the register row (a load that declines has
  written nothing); a run of consecutive 16-byte pieces is one `load_run`/`store_run`; the addresses of a full warp in `h_global`, `h_shared` and
  `h_local` are counted loops; a word load no longer goes through the sub-word extension `switch`.

### Results

`bench.py --ab` base (build of d8ec6128) against this branch, `--size full`, 15 alternating runs, one host thread, CPU 4 of server3 (load 4 to 6: the other sessions
were idle), counters identical in every row. The A/A control of the same build against itself (15 runs, the first six rows) reads 0.995x to 1.001x. An earlier
11-run pass of all 16 workloads at load 15 to 33 gave the same picture (shared 1.99x, atomics 1.75x, mem 1.38x, local 1.20x, divergent 1.18x, warp 1.16x, launch
1.10x; texture, f32_fma, f64_fma, f64_subnormal, wmma, mma_ptx, api and startup 0.984x to 1.018x, "same"; its A/A: 0.985x to 1.018x, startup 1.026x with p 0.019 on 7 ms).

| workload | base Mi/s | this branch | B/A median | best | p | verdict |
|---|---|---|---|---|---|---|
| shared | 13.89 | 27.85 | **2.01x** | 2.01x | <0.001 | faster |
| atomics | 8.14 | 14.27 | **1.75x** | 1.75x | <0.001 | faster |
| mem | 18.80 | 26.90 | **1.43x** | 1.49x | <0.001 | faster |
| local | 35.37 | 43.02 | **1.22x** | 1.21x | <0.001 | faster |
| warp | 40.18 | 47.23 | **1.18x** | 1.20x | <0.001 | faster |
| divergent | 23.57 | 27.67 | **1.17x** | 1.17x | <0.001 | faster |
| launch (k/s) | 503 | 557 | 1.11x | 1.11x | <0.001 | faster (its kernel's coalesced load and store) |
| int_alu | 86.30 | 88.41 | 1.024x | 1.024x | <0.001 | faster by the table's rule; not a function this branch changed (the first pass read 1.018x, p 0.010), take it as code placement |
| texture | 4.16 | 4.17 | 1.002x | 0.999x | 0.51 | same |

One change at a time (each against the build before it, 9 to 15 runs, the load of the day varying, so the product of these is not the table above):
chain into a diverged group `divergent` 1.096x; `lanes_at` four per movemask `divergent` +6%, `warp` 1.08x; atomics batch 1.49x; `Form::Atom` 1.074x; the alignment
mask 1.034x (p 0.008); cache-line stripes 1.057x at one thread (and 2.0x at 8 host threads, load 94, counters differ there as the CAS loop's do); shuffle
shapes `warp` 1.128x; barrier state `shared` 1.23x and `divergent` 1.028x; `Block::live` `shared` 1.047x (p 0.008); shared word runs `shared` 1.095x; direct
row runs `mem` 1.133x and `shared` 1.152x; 16-byte runs `mem` 1.183x; counted address loops `mem` 1.067x, `shared` 1.086x and 1.088x; local and shared lane
copies without the extension `switch` `local` 1.188x.

`VGPU_TIER_STATS=1` after: `divergent` 3,592,605 of 3,594,537 runs chained (was 1.25 M of 3.6 M), `atomics` 94.9% of the warp instructions in the tier (REDUX 12,800,
VOTEU 3,200 and S2UR 128 are the interpreter's, see below), the others 99.2% to 100%.

### Tests (wired into CMake)

* `test_sass_lanes` (new): `lanes_at` and `lowest_pc` against the lane loop (addresses that differ only in the low or the high half), `wait`, `waiting_on`
  and `wake` against the lane loop (whole warps and masks, other kinds and arguments among the lanes), `row_uniform`, and `shfl_move` for every shape the analysis gives
  against the per-lane rule.
* `test_sass_atomics` (new): `sassatom::apply` and every `*Step` against a frozen copy of the lambda `exec_mem` had (all operations and types, compare-and-swap, the
  edges of each type, f32 sums that are subnormal).
* `test_tier`: the form table lists `atom`; `shfl_shape` against the per-lane rule for every mode, lane 0 to 40 and every clamp and segment mask.
* The programs that run the whole engine are `bench_counters*` (results checked against the host, counters against the recorded ones), `e2e_sass_tier` (tier against
  the interpreter, `k_direct` has the atomics of every operation and type with their old values), `e2e_sass_fastpath` (the fast paths against `VGPU_FASTPATH=0`: coalesced
  and scattered loads and stores of 1 to 16 bytes, shared accesses with conflicts and broadcasts, `atomic_forms` over one address, a few, distinct and scattered ones,
  `shared_atomic_forms`) and `e2e_sass_archs`. None of the e2e programs could be built on the experiment server (no nvcc); the harness of `tools/bench` ran with the
  cached program, which checks every result and every counter against the base build.

### Mutation checks (a one-line change on the experiment server, the test named failed; reverted)

| mutation | test that failed |
|---|---|
| `lanes_at` without the high halves | `test_sass_lanes` |
| `wake` of a whole warp adds 32, not 16, to the addresses | `test_sass_lanes` |
| f32 atomic add without the flush of a subnormal result | `test_sass_atomics` (it did not fail until the operands included sums that are subnormal: the pool now has them) |
| `step_for` gives the and-step to OR | `test_sass_atomics` |
| shuffle shape Down's in-range count off by one | `test_tier`, `test_sass_lanes` |
| butterfly `pshufd` order of the xor-1 swap | `test_sass_lanes` |
| `Form::Atom` without `dst = kNoReg` | `test_tier` (3 of 8 failed before it was set) |
| chain group = all runnable lanes | bench `divergent`: `collatz thread 0` |
| atomic address hash treats a collision as the same address | bench `atomics`: `hist[0]` |
| `h_atom` drops the lowest lane | bench `atomics`: launch timeout (a CAS loop never ends) |
| atomics' stripe locks not taken (8 host threads) | bench `atomics`: `single counter` |
| `Block::live` three warps short of the block | bench `shared`: `block 0 reduction` |
| BAR arrival leaves out the lowest lane | bench `shared`: hangs (killed by the harness) |
| shared word run copies from `at + 4` | bench `shared`: `block 0 reduction` |
| shared word run counted as two passes | bench `shared`: counters differ (`shared_bank_conflicts`) |
| direct row run starts one lane late | bench `shared`: `block 0 reduction` |
| 16-byte run transposes the words | bench `mem`: `stream4[0]` |
| local lane loop stops at lane 30 | bench `local`: `thread 351` |

### Not done, and negative results

* *Pick the first lane's group first, look only at the other lanes for a lower address* (`tier_pick_group`), and an early return of `tier_release_bsync` when a member is not
  waiting at all: `divergent` 1.000x (p 0.49), `warp` 0.987x; dropped.
* *A tier form for REDUX and VOTEU* (exec_warp does not use the group, so the handler is one line): `atomics` 1.010x (p 0.51); dropped.
* *The shared word run for any contiguous prefix of lanes*: with a `memcpy` of `count` words and the lanes found by bit scan `shared` 0.891x; with counted loops
  and four-word vector copies 1.008x (p 0.17); only the full warp has the fast path.
* Earlier entries of this file say SASS atomics had nothing to gain ("the one-lookup read-modify-write", and the "direct" dispatch of the instructions that end a
  run, `atomics` 1.008x). The first was about `MemoryManager::rmw_scalar`, which the batch already avoided; the second left `exec_mem` as it was. The gain here is in
  `exec_mem`'s own lane code (the division, the scan, the probes, the locks) and, once that was cheap, in the tier form (1.074x).
* *Atomics' value rules through a template taking the step as a generic lambda* (`sassatom::dispatch`): `atomics` 0.96x (p < 0.001) with one more percent of
  instructions and three times the L1i misses (`exec_mem` split in half, the lane loops in an out-of-line `dispatch`); replaced by named functors and a `switch`, 1.004x of the
  version it was cut from. A lesson for the next one: after moving code into a header, A/B again, whatever the instruction count says.
* *Not taking the stripe locks at all* is 1.14x of `atomics` at one host thread (measured to see what is left); not a change one can make.
  A spin lock for the stripes, and one `MemoryManager::rmw_batch` instead of `load_batch` and `store_batch` (each runs `batch_ok` over the addresses: 6 to 9% of
  `atomics`) were not tried.
* `memcount::detail::sectors_fast` is 25% of `local` (a thread's private array indexed by a value of its own: the window form walks all 32 lanes); the counts have closed
  forms for strides and windows, not for this; left.
* `step_warp` itself is 45% of `divergent` and flat (the three counters, `w.steps`, the guard, the call per op): the divergent path's cost is the per-lane work of
  `tier_set_pcs` and the branch select, 64-bit addresses a lane. Keeping the counters in registers across a run would touch the loop that tier 2 hooks into.
* Sanitizers, on the experiment server: `test_tier`, `test_sass_lanes` and `test_sass_atomics` built `-fsanitize=address,undefined -O0` (Debug) and run with
  `detect_leaks=1`, `halt_on_error=1`: clean. The cached bench program run on a shim built the same way (`LD_PRELOAD` of libasan, quick size) for `divergent`, `shared`,
  `warp`, `atomics`, `mem` and `local`: all PASS, no report. The shim built `-fsanitize=thread -g -O1` (`setarch -R`, `VGPU_THREADS=4`) for `atomics`, `shared`,
  `divergent` and `mem`: PASS, no warning. Not run there: the e2e programs and their sanitizer legs (no nvcc), `test_jit` (fails on the base too: no `c++` on its PATH).

## Round sp: the four pieces together, and what they cost (integration A/B on the Zen 4 box)

The API-call, AMD-memory and SASS-control pieces above merged without a code conflict and with identical counters. A/B of the merged tree against its base (one pinned CPU, quiet machine, counters identical in every row): NVIDIA SASS `shared` 2.0x, `atomics` 1.75-1.85x, `mem` 1.47x, `local` 1.21x, `divergent` 1.20x, `warp` 1.18x, `launch` 1.14x, `api` 1.43x (SASS and PTX); AMD `buf` 2.4-2.8x, `narrow` 1.6-2.0x, `mem` 1.3-1.6x, `lds` 1.3-1.55x, `flat` 1.2-1.35x, `atomics` 1.25-1.6x, `api` about 2.9x.

Known cost, found and fixed: the f64 matrix instructions of the AMD engine ran slower in the merged tree (`mfma_f64` 0.83-0.94x, `mfma_f64_4b` 0.90-0.93x, instructions identical, cycles up 12-27%). The cause was the code, not the data: the four-trip inner loop of `mfma_tile_fma_hw` straddled a 64-byte line in the merged build. The K = 4 tile is now written out as four fused multiply-adds in the loop's order (see the launch section), after which the merged tree reads `mfma_f64` 1.25x, `mfma_f64_4b` 1.08-1.10x, `mfma` 1.03-1.06x. Residual costs on the final tree (MI300X, one pinned CPU, p <= 0.005, counters identical, same placement-sensitivity family, not chased): `smfmac` 0.96-0.97x, `mfma_i8` 0.97x; NVIDIA PTX `local` 0.96x (code none of the pieces touches).

## The cost of starting work (branch `sp-launch`)

Per-work-group and per-block set-up and the launch path, measured on server3 (one host thread, cpu 2; `perf record -e cycles:u`, then
`perf stat` for cycles, instructions and branch misses, which do not move with the load the way the harness's times do).

### Where the time went

* **AMD `launch` (one-wave launches).** `set_up_group` was 38 to 56% of the simulator's samples, and 85% of that was one loop: the wave's
  register file, `std::vector<std::array<uint32_t, 64>>(256)`, is filled by copying one 256-byte row 255 times (sixteen 16-byte
  loads and stores a row, 64 KB in all). The "dropped" entry above measured the vector in isolation at 0.7 us; in the launch loop it is the
  largest single cost (about 1.3 us a wave at 3 GHz). After a memset instead, libc's memset of the same 64 KB was still 25% of the samples.
* **NVIDIA SASS `launch`.** `Runner::init_block` 8.9%: `w = Warp{}` builds a 3 KB temporary and copies it over the warp (the register
  file and local memory are freed and allocated again for every warp), `blk.tmem = exec::TensorMemory{}` builds two `std::deque`s (four
  allocations and frees a block), and a `Block` was constructed for every launch. Per launch the Runner constructor parses the module's
  constant-bank table (a `substr` and `stoul` for each symbol, two `std::map` insertions) and looks up `".nv.constant0." + name`;
  `compute_occupancy` does about six 64-bit divisions with the same numbers every launch; `Runner::run` takes `fail_mu` and `stats_mu_` for
  every block of a launch that has one thread.

### What changed

* **AMD register file** (`VgprFile::Rows`, amd/src/gcn_exec.cpp). One block of memory, zeroed with `memset`, 64-byte aligned (a row never splits a cache line). The matrix path's four tiles (`TileBuf`, amd/src/tile_buf.hpp) start at fixed places in the page: with the rows aligned and the tiles wherever the heap put them, `mfma_f64` read 23% more cycles (2.74 G against 2.22 G, L1 misses doubled; the cause was not isolated further than the tiles' places); with the rows from `malloc` instead, `valu_f32` (rx6800) and `wave` read 3% more, `valu_f64` 1.5%. Aligned rows and placed tiles are at the base's cycles on all of them. A file of the usual 256 rows goes back to a
  per-thread pool (32 of them) with `dirty`, the number of rows that may be non-zero, and the next wave to take one clears only those. A
  work-group that ended normally (every wave `done`, no debugger) marks its files with the highest vector register any instruction the
  machine decoded names, plus 16 for the widest result, and at least v0 to v2 (`set_up_group` seeds them): `Group::~Group`. An
  instruction that indexes registers at run time (`s_set_gpr_idx_*`, `v_movrel*`, `v_swaprel*`, `s_set_vgpr_msb`) makes the bound
  "everything". The bound lives in the decode cache (`DecodeCache::vgpr_rows`) and is raised by `gcn::decode` through a per-thread sink
  the `Machine` sets for its life (`gcn::set_vgpr_rows_sink`), so the interpreter's fetch path is the code it was (see the layout
  note below). The registers stay bit-identical: a wave reads zero in every register it did before, because a row is cleared unless
  every instruction that could have written it has been decoded and names a lower register.
* **AMD per-group allocations** (`PooledAllocator`). The list node of the groups a thread holds and each group's vector of waves come from
  small per-thread free lists (one to four elements, four of each).
* **NVIDIA SASS block** (src/sass/exec.cpp). `init_block` builds each warp in place (`~Warp`, placement `Warp()`; the register file and local
  vectors keep their memory and start empty), `TensorMemory::reset()` replaces the temporary, and a thread keeps its `Block`s between
  launches (`PooledBlock`, up to four, taken for the length of a worker so a launch from inside a block takes its own). A launch on one host
  thread skips `fail_mu` and `stats_mu_`. A module's per-kernel facts (the constant banks that are backed by memory, their addresses and sizes,
  the bank 0 template, the code entry) are worked out by the first launch of the kernel and kept in the module (`Module::KernelStatics`,
  an atomic slot per kernel); the banks are arrays of 32 (a bank is named by five bits of an instruction) instead of two `std::map`s that
  `cbank32` searched for every constant read.
* **`ptx::compute_occupancy`** keeps the thread's last answer (a loop of launches of one kernel asks the same thing every time).

### Results (B/A of the medians, base d8ec6128, 9 alternating runs, p by Mann-Whitney; counters identical in every row)

| workload | mi300x 1 thread | mi300x all | rx6800 1 thread | rx6800 all |
| --- | ---: | ---: | ---: | ---: |
| launch | 1.70 | 1.65 | 1.99 | 2.13 |
| launch_wide | 2.39 | 2.39 | 2.43 | 2.40 |
| launch_mid | 2.29 | 2.31 | 2.35 | 2.35 |
| launch_sync | 1.34 | 1.35 | 1.63 | 1.62 |
| graph | 1.54 | 1.54 | 1.85 | 1.87 |
| stream | 1.03 | 1.03 (same) | 1.07 | 1.09 |
| startup, api | same | same | same | same |

p < 0.001 for every row marked faster. NVIDIA SASS: `launch` 1.26x (504 to 633 k launches/s, p < 0.001); `api`, `startup`, `int_alu`,
`mem`, `shared`, `warp`, `divergent` and the PTX engine's rows are the same (the PTX `mem` and `atomics` rows read 0.95 to 0.98 and
are the PTX engine, which this branch does not touch). `--check-counters` matches the recorded counts on 11 AMD workload/target pairs.

**Interpreter workloads and code layout.** The first versions of the register-file pool measured `valu_int`, `smem`, `lds`, `divergent` and
`wave` 3 to 6% slower with instruction counts up by under 1% and branch misses in `run_groups` (the interpreter's loop is inlined into it)
up to 60%. It was layout, not work: padding the translation unit changed nothing, but putting the decode-cache note in the fetch lambda
(then in `DecodeCache::get`, then in a noinline helper) each moved the numbers, and calling it from `Machine`'s constructor through a
thread-local sink with `gcn::decode` kept out of line (`[[gnu::noinline]]`) returned `fetch` to the code it had; `perf stat` cycles
are then the base's within 1% on `valu_int`, `smem`, `lds`, `divergent`, `wave` for both targets (for example `divergent` rx6800 2.35 G
cycles against 2.38 G, `wave` 0.757 G against 0.761 G). The A/B of the same build still reads `valu_int` rx6800 between 0.96 and 1.00
from one run to the next, and the A/A control (the base's own libraries copied to another path) read 0.95 to 1.19 on `launch` (p 0.3 to 1)
and 0.97 to 1.02 elsewhere: the harness cannot see 3% on a host this loaded (load 1 to 47 during these runs).

### Tests (wired into CMake)

* `test_amd_gcn_register_pool` (amd/tests/unit/test_gcn_register_pool.cpp): kernels written out word by word (gfx942). One writes v200 and
  ends, the next stores v200 without writing it; the pair runs on one thread, so the second gets the first's file, and has to read zero. The
  same through `s_set_gpr_idx_on` (the code never names v200). A kernel that sets and stores v200 itself reads it back. Nine to
  thirty-three work-groups of one to four waves on one thread each store their number. Mutations that fail it: the group's bound fixed at
  3 rows (2 tests fail), the `s_set_gpr_idx` name left out of `vgpr_rows_written` (the indexed test), no sink set by the machine (2), no
  clearing of the dirty rows when a file is taken (2).
* `test_sass_block_reuse` (tests/unit/test_sass_block_reuse.cpp): SASS for sm_86 written out word by word (the test checks nvdisasm's text of
  each word), loaded from a cubin it builds. Every block of a grid runs (one to 100 threads); a register a kernel sets reads zero in the next
  block, the next launch and the next kernel; a module's second kernel starts at its own code. Mutations that fail it: `init_block` not
  rebuilding the warp (2 tests), `assign` of the register file replaced by `resize` (1), the per-kernel facts shared by every kernel of every
  module (the two-kernel test).
* `test_regalloc`: `occupancy_answers_do_not_carry_over_from_the_call_before` (each of the nine numbers and the rules changed alone, right
  after the base call and before it). Mutations that fail it: the key leaving out the warp size, the key being the registers alone, the
  per-block register limit left out of the rules.
* Run on server3: Release `ctest -R test_amd` (168 tests, the same-bits, packed, memory, tier, codeobject, matrix and nan-order guards among
  them), all 699 (3 failures that are the host: no `c++` on the tests' PATH for `test_jit` and `e2e_private_loads`, no `.git` for
  `lint_curl_retries`; the `e2e_*` tests skip, there is no nvcc), ASan+UBSan on the AMD and SASS tests above, `-fsanitize=thread` on the
  register-pool, threads and atomics tests (the full sanitizer legs run in CI).

### Not done, and negative results

* The nine `getenv` calls of a SASS launch (`VGPU_FASTPATH`, `VGPU_THREADS`, `VGPU_LOCKSTEP` twice, `VGPU_TIER_*`, `VGPU_SASS_TRACE`,
  `VGPU_TC_EXACT`, `VGPU_TRACE`) are 4% of the NVIDIA `launch` workload. Not cached: tests set `VGPU_FASTPATH` and `VGPU_THREADS` between
  launches of one process. A single pass over `environ` for all of them would save about 2% and was left.
* Reusing a finished work-group's `Group` (its LDS and scratch vectors, not only the node and the waves' vector) measured the same as the
  allocator and moved the interpreter's layout; dropped.
* Sizing the register file by the kernel's declared VGPR count is still not done (see the entry above): the pool clears by what the decoded
  code names, which is exact.
* `Warp` is 3 KB of state that must start as zero (`ur[256]`, the per-lane arrays); the in-place rebuild is a memset, and is what is left of
  `init_block` (4.4% of the SASS `launch` samples). The PTX interpreter's `setup_block` (7%) was not touched.
* The remaining AMD `launch` profile is the stream and queue plumbing (`pthread_mutex_lock/unlock` 17%, `Publisher::refresh` 5%, the work
  queue thread hand-off), not set-up.

### The f64 tile and the position of its loop (merged tree)

In the merged tree (launch, memory, API pieces together) `mfma_f64` read 2.7 G cycles against the base's 2.2 G, 8% of it without
the launch piece, with identical instructions, branches, misses and tile addresses (the tiles sit at the same places in the page whatever
`TileBuf` is told, and moving them changed nothing). What moved was the code: `mfma_tile_fma_hw`'s inner loop, `add; vfmadd231sd (mem);
add; cmp; jne` for K = 4 (the f64 matrix instructions' K), is five instructions of a four-trip loop, and it ran three times as slow when
it sat across a 64-byte boundary of the code (0x...cb0 to 0x...cc3) than inside one. The loop is now written out for K = 4 (four fused
multiply-adds one after the other on one sum, the order of the loop: bit for bit the same, the existing `a_fused_tile_gives_each_output_the_chain_of_fmas`
fails if two are swapped), so there is no loop to be placed: `mfma_f64` 175 against the base's 218 (units of 10M cycles), `mfma_f64_4b`
112 against 120.

`valu_f64` (mi300x) read 1.8% more cycles in the merged tree than the pieces without launch, from a new pointer in `Group` (`cache`, for
the destructor that marks the register files) placed in the middle of the struct, before `id`, `nb_members` and the rest: the hot
loop's offsets into `Group` moved and its code with them. It is the last member now, and the cycles are the base's.

## Round sp2: the AMD ALU, scalar and control paths (branch `sp2-amdalu`)

The compute loops of the AMD tier already ran at IPC 2.5 to 5, so the gain had to come from fewer host instructions for each wave instruction. The profile
(perf on `valu_int`, `smem`, `divergent`, `lds`, `valu_f64`) found, in order of size: `v_mad_u64_u32` (one 128-bit multiply a lane, 1/3 of `valu_int` and
`smem`), the per-instruction frame of `run_round` (about 60 host instructions and a dozen dependent loads before the handler is called), handlers whose
glue (operand rows, EXEC merge, 64-bit pair gathers) ran as baseline x86-64 beside AVX2 lane kernels, `rep movs` row copies (a copy of `n * 4` bytes with a
variable `n`: GCC's expansion costs 30 cycles on Zen 4, 25% of `valu_f64`), and LDS accesses one lane at a time.

Measured on server3 (Ryzen 7 PRO 8840U, one SMT sibling, cpu 4), Release, GCC 13.3, simulated MI300X and RX 6800, `tools/bench-amd` A/B against a build
of origin/main 99b013a3, 9 alternating runs, one host thread (the all-threads rows agree), counters identical in every row:

| workload | mi300x | rx6800 |
| --- | ---: | ---: |
| `valu_int` | **2.15x** | **2.18x** |
| `valu_f32` | **1.57x** | **1.84x** |
| `valu_f64` | **3.74x** | **2.45x** |
| `smem` | **2.41x** | **1.63x** |
| `divergent` | **1.62x** | **1.68x** |
| `wave` | **1.99x** | **1.71x** |
| `lds` | 1.29x (p 0.16) | 1.21x |
| `ldsx` | 1.24x | 1.27x |
| `mfma`, `mfma_f64`, `mfma_f64_4b`, `mfma_bf16` | 1.16, 1.06, 1.07, 1.18 | not on RDNA2 |
| `mfma_i8`, `smfmac` (the residuals of round 1) | 1.01 (p 0.22), 1.07 | |
| `launch`, `api` (controls) | 1.01, 1.00 (p > 0.3) | 1.04, 1.04 |

p < 0.001 for every "faster" row not marked. The first A/B of the mfma canaries (before the last change below) read `mfma_f64` 0.96x (p 0.01) with *fewer*
instructions (8.09 G against 8.25 G) and 3% more cycles: the NaN scan that follows a matrix tile (`for each output: if (x != x)`), a 256-trip loop of six
instructions in `matrix_multiply_fast`, now sat across a 64-byte code line (0x...1b0 to 0x...1ca) where it had sat inside one. It is one branch-free pass
now (`any_nan |= out[e] != out[e]`, the fix-up loop only if any), which has no small loop to place; `mfma_f64` reads 1.06x, `smfmac` 1.07x.

### What changed

* **`v_mad_u64_u32`** (`k_mad_u64`, `gcn_tier_kernels.inc`): `vpmuludq` on zero-extended lanes, the sum with the 64-bit addend, the carry out as an unsigned compare
  (`sum < c`: a 32x32 product fits in 64 bits, so the 128-bit carry is exactly a wrapped sum), the rows split with `vpermd`. A few active lanes (8 or fewer) keep
  the per-lane loop. `valu_int` 1.44x, `smem` 1.70x from this alone.
* **Handlers built twice** (`TIER_DUAL`): the body of `h_int`, `h_f32`, `h_f64`, `h_l64`, `h_cmp`, `h_cmp64`, `h_cndmask`, `h_ds` and `h_mad_u64` is
  `always_inline` into a baseline and a `target("avx2,fma,popcnt")` function, and `Tables::avx2` picks one when the instruction is decoded (the glue helpers are
  `always_inline`, so the row fills, the EXEC merge and the NaN scans are vectorized too). `valu_f64` 1.67x, `valu_f32` 1.2x, `divergent` 1.17x.
* **The frame** (`gcn_exec.cpp`): `fetch_next` finds the next instruction through `Inst::next` (the instruction the wave ran after this one last time, checked
  against the program counter, a relaxed-acquire pointer any thread may set); the tier's `Regs` is made once per round of a wave (`Wave::tier`, cleared by
  `run_round` and where the vector register file may have moved) instead of 14 stores per instruction; the unit counter is picked by an array of pointers rather
  than a switch. `valu_int` 1.54 to 1.92x, `smem` 1.7 to 2.35x, `divergent` 1.17 to 1.4x.
* **Row-wise f64** (`k_f64r`): add, multiply and fused multiply-add on the pairs' low and high rows, eight lanes to a step (`vpunpck`/`vperm2i128` in,
  `vshufps`/`vpermq` out), scalar and constant operands as broadcasts in the kernel (24 kernels: operation x set of uniform operands), the lanes with a NaN result
  as a mask; any NaN result in an active lane gives the whole instruction to the general handler (which of two NaN payloads wins stays its business: the
  `nan_order.hpp` rules apply unchanged). `v_mov_b32`/`v_mov_b64` are row copies or fills straight into the destination; integer handlers whose sources are all
  vector registers call the kernel on the rows themselves; row copies are fixed-size moves (`copy_row`), not `rep movs`; the 64-bit compares are AVX2.
* **LDS** (`h_ds`): the lanes that are one run (all on, or a prefix as a reduction leaves them) with consecutive word (or group) addresses are one copy to or from
  the LDS (wide forms transposed through a small buffer; the two-offset forms only when their two runs are apart, since the lane order decides the bytes
  otherwise); all-lanes-on `ds_bpermute` and plain accesses are counted loops. New handlers: `ds_write2_b32` (it had none: its data operands are a register
  each, the classifier asked for a pair), `ds_read_b96`, `ds_write_b96`, `ds_add_u32` (the lane-order loop).

### Tests (CMake-wired) and the mutations that fail them

* `test_amd_gcn_tier_same_bits`: three new flavours of state (18 and 19: an LDS access whose lanes have consecutive addresses, all on or one run of lanes on; 20: the
  integers' edges, so carries and borrows happen) and, new, its twin under `VGPU_SIMD=base` (`..._simd_base`, the baseline build of every kernel and handler).
* Mutations, each built and run on server3 (the tier test restricted to the instruction concerned, or the named kernel tests):

| # | mutation | result |
| --- | --- | --- |
| M1 | `k_mad_u64` carry compare with the operands swapped | `v_mad_u64_u32` differs (520 runs) |
| M2 | `h_ds` contiguity test dropped | every `ds_read*`/`ds_write*` differs |
| M3 | `ds_add_u32` stores the register instead of adding | `ds_add_u32` differs (144) |
| M4 | `store64` writes the words swapped | `v_lshrrev_b64` differs (425) |
| M5 | 64-bit compare `<=` done as `<` in the AVX2 kernel | `v_cmp_le_i64` differs |
| M6 | `k_f64r` fused add takes the wrong addend | `v_fma_f64` differs (1276) |
| M7 | `fetch_next` takes the hint without the program-counter check | `test_amd_gcn_exec_*` and `test_amd_gcn_lds_*` fail on all targets (the instruction-level guards cannot see it) |
| M8 | unit counter of a tier instruction swapped | `test_amd_gcn_same_bits`, `test_amd_gcn_counters` fail |
| M9 | b64 read de-transposed in the wrong order (earlier tree) / write2 overlap guard dropped | `ds_read_b64`, `ds_write2_b32` differ |

  Not shown by a mutation: the clearing of `Wave::tier.ok` at the start of a round (a defensive reset: a wave is made fresh for each group here, so nothing
  observes it).

### Not done, and negative results

* `s_load_dwordx*` as one load: dropped in round 1 (1.015x). What dominates `smem` is the 21 scalar ALU instructions around two scalar loads: the frame above.
* A `fetch` hint for two-way branches (one hint per instruction thrashes only on alternating branches; no workload showed a miss rate worth a second slot).
* Scalar handlers by operand shape (SGPR/constant/pair variants without the `rd`/`wr` switch): written as a design, not measured, not kept.

## The PTX interpreter's memory, atomics, shuffles and diverged warps (branch `sp2-ptxpath`, base origin/main 99b013a3)

* Coalesced global loads and stores of a run of lanes move with `MemoryManager::load_run`/`store_run` (registers' rows straight to and from memory); consecutive shared accesses are one block copy; the lane loops of shared and local gathers have the element size as a template constant; address generation is a counted loop and a coalesced run is counted by `memcount::run_sectors`.
* A warp's PTX atomics on device memory go through one `rmw_batch` call (new `always_store` flag so resident chunks match `rmw_scalar`), only on a one-thread launch and after `sassatom::batchable`.
* `micro_shfl32` with one lane/clamp for all lanes of a full warp runs without the per-lane mode switch.
* A diverged warp's micro instructions run in batches while `select_path` would pick the same path again (merge, lower pc, starvation emulated per instruction; boosted paths not batched).
* Tests: `test_ptx_mem_runs`, `test_ptx_paths` (fast path against `VGPU_FASTPATH=0`, memory, registers, every counter, errors).
* Results against origin/main (B/A median, 9 runs, cpu 0, counters identical): mem 1.36x, shared 1.60x, atomics 1.39x, divergent 1.20x, warp 1.06x, launch 1.17x (not a change of this branch's launch path; take as placement), int_alu/f32_fma/f64_fma/local/api same (0.99-1.04, p > 0.25).
* Not done: a micro form for ld/st (skips step/dispatch; est. 10-20% on mem/shared), the Interpreter constructor's 6.5 KB of zeroed scratch (5% of a PTX launch), `setup_block` reuse, atomics on multi-threaded launches (still per-lane stripe locks). The "unexplained" PTX int_alu/local 0.96-0.97x residuals were not isolated.

## Round sp2: the SASS tier's stretches, the group pick, LEA, the environment pass (branch `sp2-sassrest`, base origin/main 99b013a3)

* **Stretches** (`MOp::run`, `tier_issue`): a run of straight, unguarded, register-only instructions (`tier::pure_register_form`) is issued back to back for the whole group: the store gate, the step count, the guard and the three counters are done once for the stretch (not under tier 2's hook loop, nor with `VGPU_TIER_STATS`). **Direct writes** (`tier_out`): the lane-by-lane kernels write a full warp's result into the destination row; the row copy of `tier_store` is gone (a partial warp still blends from a temporary).
* **Group pick** (`sasslanes::pick_group`, `src/simd/lanes.cpp`): a converged warp is found as before; a diverged one gets its lowest address and its lanes in one AVX2 pass instead of lanes_at + lowest_pc + lanes_at.
* **LEA** (`include/vgpu/exec/sass_lea.hpp`): four lanes to a vector, the carry in and out as masks (the 64-bit address pair of every pointer calculation).
* **Environment**: one pass over `environ` for the seven variables a SASS launch reads (`launch_env.hpp`); read at every launch as before.
* **Atomics**: `MemoryManager::batch_window/batch_load/batch_store` do the checks of load_batch and store_batch once; `sassatom::batchable` still runs before any stripe lock.

A/B against origin/main (cpu 6, 9 alternating runs, counters identical): int_alu 1.56x, f32_fma 1.56x, warp 1.20x, local 1.15x, launch 1.10x, mem 1.10x, divergent 1.09x, f64_subnormal 1.09x, f64_fma 1.07x, atomics 1.04x; shared, api, texture same; PTX int_alu/f32_fma/mem/atomics/local/launch/api same (0.98-1.02, p > 0.1).

Not done / negative: pc and wait stores in 16-byte blends (halved store-forwarding stalls, `ls_bad_status2.stli_other` 50M to 22M, but +1.3M branch misses; divergent cycles unchanged, dropped); a spin lock for the stripes (the PTX engine and host atomics share the mutexes: not changeable here); `VGPU_TRACE` getenv in launch_kernel_impl (1%) and the runtime's lockstep_mode (0.9%) are still per launch; texture (sampler in interpreter.cpp, ~450 cycles a lane-fetch, spread over fetch_texel/tex_finish/footprint) and local's `sectors_fast` (28% of `local`) not touched; f64 is dominated by the DFMA kernel (47%).

## Tensor cores: float products, tile kernels, wgmma, tcgen05 and the other mma forms (branch `sp2-nvtensor`, base origin/main 99b013a3)

Measured on server3 (Ryzen 7 PRO 8840U, cpu 2 alone, 9 to 15 alternating runs, GCC 13.3), `tools/bench/bench.py --ab` against a build of origin/main; counters
identical in every row. Profile first: SASS and PTX `wmma`/`mma_ptx` spent 66 to 78% in `tc86::mma_avx2_chunked`, 8 to 11% in `hmma_k`; the kernel itself
executed 12,683 instructions per m16n8k16 call (about 400 per row-chunk, where about 140 are needed): the lambda took N, K and the arrays through its
closure, which the may-alias stores re-read after every store, the row loops stayed rolled (arrays in memory) and the colmask was a read-modify-write chain.
The forms the benchmark does not run were far worse: `wgmma` m64n256k16 111 us a warp, int8 `mma.sync` 7.5 us, `.b1` 41 us, fp8 7.7 us, sparse f16 10.7 us
(every operand decoded through a `std::vector<double>` and a switch, summed by scalar double loops).

### What changed
* **sm_86 sum, float products** (`mma_avx2_chunked_f`, include/vgpu/exec/tc_ampere.hpp). A product of two f16, bf16 or tf32 operands is exact in f32 when its exponent
  is in range, so the aligned term (the product's magnitude shifted right, bits dropped) is `cvttps(a * b * 2^-g)`: two multiplies and a truncating convert
  instead of an integer multiply, two subtractions, abs, variable shift and sign. Taken when the exponent fields say every product is a normal f32 (smallest sum
  of nonzero exponent fields >= 128, largest <= 379, no float subnormal with a significand, no infinity or NaN anywhere); else the integer kernel runs, as it does
  for a chunk whose 2^-g is not a float (a lane finished by the scalar rule). Addresses hoisted into locals, rows unrolled (`VGPU_TC86_UNROLL`), eight rows at a
  time (`VGPU_TC86_ROWS_F`). A loop of `tc86::mma` m16n8k16 f16: 2,680 -> 2,015 cycles, 12,683 -> 8,630 instructions (B/A 0.75; bf16 0.76, tf32 0.90).
* **Tile kernels** (`simd::Kernels::tile_f32`, `tile_i32`, `tile_f64`; baseline and AVX2 twins, strided C and D): the tensor-core sums over a tile, k in order, products
  rounded then added (never fused), s32 with `.sat` clamped per block, f64 fused in the host rounding mode.
* **Operand code** (include/vgpu/exec/tile_mma.hpp, src/exec/tile_mma.cpp): shared-memory matrices read per 16-byte run into float or int32 tiles, accumulator
  registers <-> tiles by shuffles, mma fragments of f16, bf16, tf32, e4m3, e5m2, s8, u8, s4, u4 (dense and sparse), f64 and `.b1`.
* **PTX `wgmma`** (`exec_wgmma_fast`) and **SASS HGMMA/IGMMA/QGMMA** (`Runner::gmma_fast`): the first warp of the group reads A and B into a float/int32 snapshot, each warp
  makes one kernel call. **PTX `mma.sync`** (`exec_mma_tiles`) and **SASS `exec_mma`** for sparse f16/bf16/tf32, fp8, s8/u8/s4/u4 (also `.sp`, `.sat`), f64 (all
  shapes, all four rounding modes; a NaN in a tile declines, the payload a fused multiply-add keeps depends on its operand order) and `.b1`; the SASS tier's
  dense `IMMA` handler through the same code. **`tcgen05.mma`** reads A and B whole from the CTA's shared memory (`Tcgen05Host::smem_direct`) and, for dense f32/s32 D in
  layouts D and A, makes one tile product of the CTA's rows.
* The new hot-path functions are in `.text.unlikely`: without that PTX `local` read 0.965 to 0.975x with identical instructions (placement); with it 0.985 to 0.998.

### Results (B/A of medians; microbenchmarks are cycles of the whole program, so startup is included and the ratios understate)
| row | B/A | p | | row | B/A |
|---|---|---|---|---|---|
| wmma sass | 1.19x | <0.001 | | PTX wgmma m64n256k16 f16 | 0.19 (111 -> 15 us a warp) |
| wmma ptx | 1.27x | <0.001 | | wgmma n128 bf16 / tf32 / e4m3 / s8 | 0.10 / 0.14 / 0.13 / 0.12 |
| mma_ptx sass | 1.20x | <0.001 | | mma.sync s8 k32 / s4 k64 / b1 k256 | 0.20 / 0.16 / 0.10 |
| mma_ptx ptx | 1.12x | <0.001 | | f64 m8n8k4 / e4m3 k32 / sparse f16 k32 | 0.38 / 0.60 / 0.89 |
| tcgen05 m128n256k16 (loop of 100) | f16 280 -> 97 us, tf32 300 -> 90, fp8 440 -> 190, s8 800 -> 110 | | SASS IMMA 16816/16832 s8, .SAT | 1065 -> 670, 1456 -> 1057, 2631 -> 1104 ns |

Canaries (sass / ptx, 11 runs): int_alu 1.00/0.99, f32_fma 1.01/1.00, f64_fma 1.01/1.01, mem 0.96 (p 0.95)/1.01, shared 0.99/1.00, atomics 0.99/1.00, local 0.99/0.985,
divergent 0.99/1.00, warp 0.99/1.00, launch 0.970 (p 0.008; perf stat: 223.6 vs 224.7 M cycles, instructions equal)/0.99, api 1.01/1.01; A/A noise 1 to 3%.

### Tests (wired in CMake) and mutation checks
`test_tile_mma` (kernels at both levels against scalar loops incl. strides, saturation, rounding modes; operand reads; layouts), `test_wgmma_fast` (PTX wgmma, every dtype, layouts,
swizzles, scales, sparse, `.satfinite`, per-lane scale-d, against `VGPU_FASTPATH=0`), `test_mma_tiles` (40 mma.sync forms), `test_tcgen05_fast`, `test_sass_mma_tiles`
(every HMMA/IMMA/BMMA/DMMA/QMMA/HGMMA/IGMMA/QGMMA word of the sm_80/86/89/90a disassembler corpora in a hand-built cubin, run with and without `VGPU_SIMD=off`),
`test_tc_ampere_fast` (float-kernel edges: exponent thresholds, subnormal operands, tf32 tails, infinities). 72 one-line mutants (a `MUT(n)` switch compiled into the
mutated unit, run per id): 72 killed; the first campaign let 7 survive (f64 forms always declined by NaNs in the data, constant sparsity selector, dead special-lane
code in the float kernel), fixed in the tests or removed. Run: Release guards, `lint_*` (the `lint_curl_retries` failure needs a git checkout), ASan+UBSan -O0 and TSan -O1
of the six new tests, `VGPU_SIMD=base` runs.

### Not done, and negative results
* SASS bench rows cannot reach IMMA/HGMMA (no nvcc here): their numbers come from `test_sass_mma_tiles`' cubins (24 blocks, 200 repeats).
* Fragment decode is still a switch per element (IMMA 670 ns, sparse HMMA 10 us a warp: `mma_sparse_column` copies the metadata into a `Lanes` per element); next step.
* tcgen05: f16 D, layout F, parts of N (layout B), sparse and block-scaled forms keep the row loop; PTX `wmma` generic shapes, `m8n8k4` f16 untouched; no AVX-512 kernel.
* `vpmaddwd` for the integer products and a float accumulator decode were not tried (Zen 4 has one-uop `vpmulld`).

## Round sp3: texture sampling, local-memory sector counts, PTX f64 (branch `sp3-texlocal`, base 03d92bd0)

### What changed
* **Texture fetch** (`src/exec/interpreter.cpp`, `src/sass/exec_ops.inc`): the sampler's cost was mostly zeroing. A 280-byte `TextureDesc local` (and, after a first fix, the storage
  of a `std::optional<TextureDesc>`, which GCC also zeroes) was `rep stos` in front of every fetch; the 8 `TexTerm`s and the sRGB code table of `tex_finish` were zeroed too; and each
  lane built a fresh 80-byte `TexFetch` (another `rep stos`, whose start-up is tens of cycles on Zen 4). The descriptor is now copy-constructed into raw storage when a fetch changes it,
  `TexTerm` has no member initializers (every footprint function sets all members), and both engines build one `TexFetch` per instruction and set only what varies per lane. Two
  fast-path forms: a linear fetch at a texel's centre (all three 8-bit fractions 0) skips the weight split, and a one-texel float fetch with 32-bit channels (`tex_finish_one_f32`)
  returns the texel as the blend would (finite as is, subnormal as zero of its sign, NaN as 0x7FFFFFFF, infinity as is). Both are skipped under `VGPU_FASTPATH=0`, which is the oracle.
* **Local-memory sectors**: `memcount::detail::sectors_fast` tries an AVX2 window kernel (`simd::sector_window_kernel`, `src/simd/simd.cpp`) first for warps whose lanes lie within 63 sectors:
  four lanes a step, each lane's first and last sector as bits of one word, popcount. Hosts and levels without AVX2 (`VGPU_SIMD=base`) keep the scalar forms; the count is the same.
* **PTX f64**: `fma.f64`, its contracted subtracts, and `add/sub/mul/div.f64` run four lanes a step (kernels in `src/simd/simd.cpp`, level-gated like the other SIMD kernels). A full warp's eight
  fma results are checked for a NaN together and stored after every operand is read; a group with a NaN result runs lane by lane under the old NaN rule. SASS f64 was already AVX2
  (`dop_avx2_t`); its remaining cost is the unpack/shuffle of lo/hi planes and the per-step NaN branch, in `src/simd/kernels.inc`, which the JIT embeds: not touched.

### Results (B/A of medians, 11 alternating runs, base /tmp/sp2-lead/build, counters identical; noise from A/A 1 to 3%)
| row | sass | ptx |
|---|---|---|
| texture | 1.59x | 1.56x |
| local | 1.20x | 1.03x (cycles 1.034, instructions -8.5%) |
| f64_fma | 0.99x (unchanged) | 1.46x (cycles) |
| f64_subnormal | 1.01x | 1.40x (cycles) |

### Tests (wired in CMake) and mutations
`linear_texture_fetches_at_texel_centres_match_the_general_path` and `f64_arithmetic_random_bits_match_the_general_path` (random bit patterns, all lanes, both guards and a diverged warp) in `test_fastpath`;
`closed_forms_private_arrays_with_and_without_the_simd_window` in `test_memcount_forms` (also run as `test_memcount_forms_simd_base`). Mutants killed: window kernel without the second sector, one-texel flush removed,
centre weight 255, contracted-subtract negation swapped, sub computed as add, NaN fallback disabled, NaN check of the full-warp path disabled, `TexFetch::layer` not set per lane (`test_exec3` layered fetches).

### Not done, and negative results
* The SASS `exec_tex` hoist of `TexFetch` has no unit test here (no SASS texture cubin without nvcc); the bench `texture` self-check and the CI e2e texture tests cover it.
* Sampler left: `MemoryManager::load_fields` 17% and `tex_footprint_linear` 17% of `texture`; the former is the allocation lookup chain (TLS, cache entry, chunk table), shared with every load.
* `local`: `h_local` and the PTX `exec_ld`/`exec_st` are 32 separate-line loads and stores per access (per-lane layout), no closed form for the data.
* A vector kernel for AVX-512 was not written (Zen 4 runs it as two halves); the f64 and window kernels are AVX2.
* Placement: PTX `mma_ptx` (the `tc86::mma_avx2_chunked_f` loop) read 0.94x with equal instructions on this tree; it moves with any change before it in the library, recheck after merging.
## Round sp3: the launch path, the API calls and startup (branch `sp3-launchrt`, base 03d92bd0)

* **One pass over the environment per launch** (`include/vgpu/exec/launch_env.hpp`, `src/exec/launch_env.cpp`): `LaunchEnv` now holds the 16 variables a launch reads (the runtime's `VGPU_TRACE`, `VGPU_LOCKSTEP`, the PTX interpreter's strict/race/fastpath/max-steps/scheduler/threads/fail-injection ones, the SASS ones). `launch_kernel_impl` reads it once and hands it on in `LaunchConfig::env`; `Device::run_kernel`, `exec::launch`, `launch_grid` and the SASS `Runner` use it, and read their own when none was given (tests, other callers). Still read at every launch.
* **PTX launch**: a thread keeps up to four `BlockState`s (`PooledBlock`; shared memory, barriers, tensor memory, the warps' register vectors) between blocks and launches; `BlockState::reset()` and `setup_block` give a block the state of a new one. The Interpreter's 6 KB of scratch lanes is no longer zeroed (each lane is written before it is read; the zero rows stay zero). `ParamBuffer` has no per-launch name table (a scan on the rare name lookup). Loads and stores skip `step()`/`dispatch()` (`step_mem`, only with the fast paths on). `Device::run_kernel` no longer reads the clock twice for a value nobody used.
* **Startup**: the two 65536-entry half tables are `constexpr` (checked against the binary16 format by a `static_assert`) and the F16C check is made on first use; `EntryFn`/`Instr` move instead of copy when a vector grows (`noexcept` on `LazyByte`/`PollsFlag` copies, `static_assert`ed); the RDNA decode tables are made per generation at first use, not all four.
* **AMD**: `hipDeviceGetAttribute` and the PCI bus calls answer from properties made once per device (`State::device_props`); the HIP calls a profiler may time read the clock for their start only when one is attached (`profile_clock`).
* Tests (CMake-wired): `test_launch_env` (new variables), `test_ptx_block_reuse` (a block starts from nothing after other kernels on the thread; the environment pass given or not; parameters by name and position), `test_amd_hip_props_cache`, `test_amd_hip_profile_clock`; existing `test_amd_gcn`, `test_exec` cover the rest. Mutations that fail them: `case 'M'` dropped (test_launch_env); `bars`, `clock` not reset, shared memory not zeroed, the given pass ignored (test_ptx_block_reuse); a NaN table entry changed and `noexcept` removed (the `static_assert`s, build fails); `step_mem` without the load count (test_exec); `ParamBuffer::find` returning the wrong slot (test_ptx_block_reuse); the cache keyed by device 0 (test_amd_hip_props_cache); the clock read always (test_amd_hip_profile_clock: 160 reads unwatched against 80); RDNA2 given RDNA3's table (test_amd_gcn).
* Results (B/A of medians, 9 runs, cpu 6, counters identical; A/A 0.99 to 1.01): PTX `launch` 1.35x, SASS `launch` 1.04x, `startup` SASS 1.12x and PTX 1.23x, PTX `mem` 1.05x, `shared` 1.09x, `atomics` 1.08x, `f32_fma` 1.06x, `mma_ptx` 1.07x, `f64_fma` 1.035x; AMD `apicheap` 3.7 to 4.0x, `startup` 1.11 to 1.17x (MI300X) and 1.40x (RX 6800), other AMD rows and canaries same.
* Placement, not work (instructions equal, cycles up): `api` 0.98x (both engines), SASS `mma_ptx` 0.97x, `warp` 0.975x, `wmma` 0.92 to 0.98x between runs (`mma_avx2_chunked_f` moved from 48 to 0 mod 64; IPC 4.03 to 3.70), AMD `smem` mi300x 0.98x (cycles +1 to 2%). Re-check after merging with the other round-3 branches.
* Not done: hidden visibility or `-Bsymbolic` to cut the 25% of SASS startup that is ld.so symbol hashing (the shims rely on one exported copy of the profiling readers); skipping `kernel_resources` for kernels that do not poll (about 2%); reusing the kernarg segment across AMD launches (changes allocation addresses); the SASS `unsupported()` pre-decode (20% of SASS startup, decides PTX fallback); a full micro form for ld/st beyond `step_mem`; merging the AMD queue's take/complete locks; telemetry's clock read per update (needed for its timestamp). The PTX parser (85% of PTX startup) is a 4000-line string chain: untouched.
* Run: Release ctest of the PTX/SASS/AMD/cupti/profiling suites (all pass; `lint_curl_retries` fails only for want of `.git` on server3), ASan+UBSan -O0 and TSan -O1 on the touched unit tests (pass).
## Round sp3: tensor cores, the leftovers (branch `sp3-tensor2`, base 03d92bd0 + sp2-lead)

Measured on server3 (cpu 2 alone, user cycles by `perf_event_open`, `nvidia_tensor_bench`: the cheapest of 5 launches of R and 2R instructions, the difference over R;
A = a build of the base tree, B = this branch; 5 to 11 alternating runs; A/A noise 1 to 2% in cycles). Counters identical in every bench.py row.

* **Fragment decode** (`tile_mma.cpp`): one instantiation per element type; dense A/B are 16-byte runs copied into a raw tile and decoded in one pass (B transposed), the
  sparse A places each lane's stored elements from the metadata nibble with every constant known at compile time (no `Lanes` copy); `mma_frag_a` takes `negate` (wgmma's
  imm-scale-a) and `wgmma_sparse_expand` spreads decoded sparse rows from shared memory. Operand registers are gathered and scattered as whole 32-lane words
  (`gather_words`/`scatter_words`) instead of widened to 64 bits and back.
* **tcgen05**: f16 D, layouts F, B, G (`.ws`), A in Tensor Memory, sparse A and the block-scaled kinds (mxf8f6f4, mxf4, mxf4nvf4, UE8M0/UE4M3 factors, one scaled B per lane
  partition with differing factors) through the tile kernels. **wmma.mma** generic shapes/types (m32n8k16, m8n32k16, s8/u8/s4/u4, f64, f16/f32 accumulators) and **mma m8n8k4 f16**
  through them too. `tile_f32c` (new kernel entry): the f32 tile with canonical NaNs in the same pass (no separate pass in the wgmma/mma/tcgen05 routes).
  `wgmma_smem_offset` is inline (a swizzled row is a power of two: shifts for divisions).
* **AVX-512 level** (`Level::Avx512`, `src/simd/kernels512.inc`, off by default; `VGPU_SIMD=avx512`): tile_f32/f32c, tile_i32 (VNNI `vpdpwssd` from 64 columns, only when every operand is a 16-bit
  number, else the multiplier form; odd `.sat` blocks too) and tile_f64 (fused, host rounding mode). `level()` still names Base/Avx2 (the JIT reads it); `kernels().level` names the table.
  Kernel microbenchmark on Zen 4 (cycles/call, AVX2 -> AVX-512): f32 16x256x16 7797 -> 7277, 16x256x64 26539 -> 25867 (-3 to -7%: Zen 4 runs 512-bit as two 256-bit halves);
  s32 16x256x32 19654 -> 10801, 16x128x32 7578 -> 5396 (VNNI); 16x8x32 507 -> 523 (VNNI packing costs 760 ns below 64 columns, so those stay on the multiplier form);
  f64 8x8x4 131 -> 69, 16x8x16 821 -> 301. End to end (same binary, `VGPU_SIMD=avx512` against default): tcgen05 s8 0.63x, wgmma s8 0.89x, wgmma f16/tf32 0.94-0.95x, wgmma e4m3 1.02x,
  tcgen05 sparse 1.04x, the rest 0.97-1.03x. Left off by default (mixed, and on this box only the int8 rows clearly gain); a CPU with native 512-bit units should gain more.

### Results (B/A of median user cycles per warp instruction, `nvidia_tensor_bench`; every p <= 0.01 unless the row says same)
| row | B/A | row | B/A |
|---|---|---|---|
| PTX mma f16/bf16/tf32 m16n8k8/k16, f32 acc, a10 | 0.76 to 0.85 | PTX mma f16/bf16/tf32 (h100) | 0.68 to 0.76 |
| PTX mma s8 k16/k32/k32 sat | 0.55 / 0.54 / 0.56 | PTX mma s4 k64 / e4m3 k32 | 0.84 / 0.24 |
| PTX mma f16 sparse k32 / s8 sparse k64 | 0.15 / 0.11 | PTX mma m8n8k4 f16 | 0.86 |
| PTX wmma m32n8k16, m8n32k16 f16 | 0.24 | PTX wmma s8 m16n16k16 / f64 m8n8k4 | 0.22 / 0.41 |
| PTX wgmma m64n128k16 f16 / bf16 / tf32 / e4m3 / s8 | 0.57 / 0.59 / 0.56 / 0.68 / 0.65 | wgmma m64n256k16 f16 | 0.61 |
| tcgen05 m128n256k16 f16 f32 D / f16 D | 0.89 / 0.18 | tcgen05 layout F / layout B (ws) / sparse | 0.55 / 0.61 / 0.54 |
| tcgen05 mxf8f6f4 m128n128k32 block-scaled | 0.04 (4.79 M -> 0.21 M cycles) | tcgen05 tf32 / e4m3 / s8 | 0.81 / 0.95 / 0.97 (same) |
| SASS HMMA.SP.16832 | 0.16 | SASS QMMA e4m3 | 0.26 |
| SASS IMMA.16816 / 16832 s8 | 0.78 / 0.78 | SASS HMMA dense 16816, 1688, IMMA s4 | 0.99 to 1.03 (unchanged code) |
Canaries (bench.py, 11 runs, sass/ptx, counters identical): mma_ptx ptx 1.16x; wmma 1.04/1.01 (same); int_alu 0.99/0.99; f32_fma 1.00/0.99; f64_fma 1.00/1.01; mem 0.975/1.017;
shared 1.01/0.99; atomics 0.99/0.975 (p 0.001); local 0.99/1.00; divergent 1.00/1.01; warp 0.99/1.02; launch 1.04/1.02; api 0.97/0.98 (p <= 0.01; api, atomics touch no code of this branch:
the hot text moved with the size of earlier objects -- see below).

### Tests (CMake-wired) and mutation checks
`test_frag_decode` (+ `_simd_base`: every type, dense/sparse, selectors 0-3, negate, and `wgmma_sparse_expand`, against the element-at-a-time decode frozen in the test),
`test_tcgen05_tiles` (240 forms: kinds x layouts x sparse/A in Tensor Memory/scale-d/block scaling against `VGPU_FASTPATH=0`), `test_wmma_generic_fast` (wmma shapes/types and mma.m8n8k4),
`test_tile_mma` (tile_f32c, wrapping s32 on operands of any size, odd/short `.sat` blocks, 16-bit boundaries), the avx512 twins `test_{mma_fast,mma_tiles,wgmma_fast,tcgen05_tiles,wmma_generic_fast,tile_mma}_simd_avx512`;
`test_wgmma_fast` now also negates A in registers. Mutants (a one-line edit of the tree, the tests run, killed = a test fails): decode 18/18 killed, tcgen05 route 22/22 killed (test_blackwell's card answers
kill the scale-table mutants the fast-vs-slow comparison cannot see, both routes share them), gather/scatter + wgmma 5/5, avx512 kernels 14/14 (after adding second-of-a-pair operands outside 16 bits),
wmma generic/m8n8k4/f32c 15 of 19: the four survivors are equivalent or common to both routes (a NaN in f64 C only, the m8n8k4 A/D layouts which fast and general share; the tf32 mutant sits in code
removed afterwards: tf32 has no generic shape). A pure dispatch bug (tc86::mma sent AVX-512 to the scalar sum, 7x slower) has no bit-level test; it was found by the bench.
Run: Release guards (amd same_bits/tier/matrix/mfma/nan_order and their `_simd_base` twins, `lint_*` but `lint_curl_retries` which needs a git checkout), ASan+UBSan -O0 and TSan -O1 (`setarch -R`) of the 11 touched unit tests
including test_fastpath: clean. Not run: e2e_* (no nvcc on server3), e2e_host_atomics (no atomic or lock code touched).

### Not done, and negative results
* AVX-512 as default: mixed on Zen 4 (above). VNNI below 64 columns: 4x slower than the multiplier form (packing per call).
* SASS/PTX `s4` decode is still scalar per nibble (IMMA 16864 unchanged, 4.6 K cycles); fp8 decode is a 256-entry table per element (QMMA 3.1 K cycles).
* wmma.load/store generic shapes are one `load_routed` per element (not touched: counters and race hooks).
* Placement: adding code anywhere before the interpreters moved their hot functions by 0x240-0x260 bytes and cost 3 to 10% on unchanged mma rows with identical instructions (misses/branch-misses moved);
  64-byte alignment of the chunked kernels did not remove it (dropped); keeping the new decode templates and SASS `gmma_fast` in `.text.unlikely` did (f64 mma 1.09 -> 0.97, SASS mem 0.95 -> 0.975).
## Round sp3: the AMD matrix path, scalar handlers by shape and the tier's no-ops (branch `sp3-amdscalar`, base 03d92bd0 + sp2-lead 2fd5a519)

**Measured first** (perf `cycles:u`, one host thread, cpu 4 of server3, the bench-amd binaries of every target, `VGPU_THREADS=1`): in every matrix row `matrix_multiply_fast` (and `wmma_fast`) itself was 50 to 85% of the time and the tile kernels 8 to 25%: what dominated was the gather and scatter around the tile, not the arithmetic. `mfma_i8` was 85% decode; the C gather was a scalar store loop whose result the tile kernel then read as 32 bytes (GCC also built `movhps` to the stack and `movaps` back for it: a store-forwarding stall on 55% of one workload's samples). The other targets' binaries (gfx950, gfx1100, gfx1201) were compiled on server2 with the harness's flags and keyed as the harness keys them.

### What changed
* **C and D as runs** (`MatrixPlan::runs`, `run_*` helpers in `gcn_exec.cpp`): output r of the n lanes of a lane group is n consecutive elements of the tile, so C is read and D written as rows of 4 with SSE2 intrinsics (the compiler's own loops were the stall above); the old per-lane tables stay for any shape whose outputs are not runs (checked when the plan is built).
* **Decode**: binary16, bfloat16, float and xf32 operands eight lanes at a time under AVX2 (`half8_to_double`: F16C, exact, a signalling NaN made quiet as `half_bits_to_double` does; gate `mfma_have_avx2()` + `f16c`, so `VGPU_SIMD=base` keeps the SSE2 path), A by 4x4 transposes and 32-byte stores, B in 32-byte rows; the 8-bit integer tiles widened in registers (`matrix_decode_a_i8` / `_b_i8`, SSE2); the double-precision tiles by rows (`matrix_decode_f64_*`); no per-instruction 64-lane identity loops.
* **Tile kernels**: `mfma_tile_f_avx2` returns whether an output is a NaN (compared while the sums are in registers: the separate 256-element scan is gone); `mfma_tile_fma_avx2` (four columns a fused multiply-add; the fused result is exact, the width cannot show).
* **WMMA** (`wmma_fast`): rows of D and C as runs of 16 lanes, per-architecture place tables (`wmma_place`) feeding the same decoders, integer C and D by rows (the sum wraps; CLAMP keeps its scalar loop).
* **Tier** (`gcn_tier.cpp`): the scalar handlers of `TIER_S2` are also built for one-word scalar registers and constants (`s2_variants`, picked at decode), `s_cmp` per comparison and shape; `buffer_wbl2/inv/gl0_inv/gl1_inv...` (a fence and a count, `mach` 11), RDNA's `s_waitcnt_vscnt/vmcnt/expcnt/lgkmcnt`, `s_version` and the `s_wait_*` of RDNA4 as no-ops.

### Results (B/A of medians, 9 alternating runs, one host thread, counters identical in every row; A = the sp2-lead build)
| workload | mi300x | rx6800 | other |
| --- | ---: | ---: | --- |
| `mfma` | **1.81x** | | |
| `mfma_bf16` | **1.71x** | | |
| `mfma_i8` | **3.46x** | | |
| `mfma_f64` / `mfma_f64_4b` | **1.98x / 1.94x** | | |
| `smfmac`, `mfma_fp8` | 1.23x, 1.24x | | |
| `mfma_f8f6f4`, `mfma_scale` (gfx950) | | | 1.03x, 1.04x |
| `wmma` / `wmma_iu8` | | | gfx1100 **3.54x / 5.85x**, gfx1201 **3.48x / 7.55x** |
| canaries `valu_int`, `valu_f32`, `valu_f64` | 0.99, 1.01, 1.03 | 1.02, 1.02, 1.01 | |
| `wave`, `lds`, `divergent` | 1.03, 1.00, 1.02 | 1.03, 1.12, 1.01 | |
| `smem` (scalar handlers) | 1.09 (p 0.01) | 1.08 (p < 0.001) | |
| `launch`, `api` | 1.03, 1.00 | 0.98 (p 0.6), 0.99 | perf stat cycles equal within noise |
Whole-process `perf stat` (cycles, instructions) on `mfma`: 1.80 G / 9.5 G to 0.97 G / 4.6 G.

### Tests and mutations
`test_amd_matrix_fast` (every matrix instruction of the corpora, tile path against the element-at-a-time reference, shuffled registers, all flavours), `test_amd_mfma_math` (new: every binary16 through `half8_to_double` against `half_bits_to_double`), `test_amd_gcn_tier_same_bits` (covers each new handler: every corpus shape with the tier off and on). Mutations (each built and run on server3, then reverted; all fail the named test):
| # | mutation | result |
| --- | --- | --- |
| M1 | C gather's high pair of a 4-run taken from the low | `test_amd_matrix_fast` fails |
| M2 | integer B's even/odd pair shuffle swapped | `test_amd_matrix_fast` fails |
| M3 | `mfma_tile_fma_avx2` K=4 reads A's element 2 for 1 | `test_amd_matrix_fast` fails |
| M4 | `half8_to_double` lane permute wrong | `test_amd_mfma_math` fails |
| M5 | the tile kernel never reports a NaN | `test_amd_matrix_fast` fails (dropping only the last accumulator's check survives: another output of the tile is a NaN as well) |
| M6 | D's high word of a double written to the low row | `test_amd_matrix_fast` fails |
| M7 | WMMA integer D wraps where CLAMP is set | `test_amd_matrix_fast` fails |
| M8 | `s_cmp_lt_u32` as `<=` | tier digest (`VGPU_TIER_ONLY=s_cmp_lt_u32`) fails |
| M9 | the fence does not count a vector memory instruction | tier digest (`buffer_gl0_inv`) fails |
| M10 | the register-register scalar read off by one | tier digest (`s_add_u32`) fails |

Sanitizers: ASan+UBSan -O0 (`test_amd_mfma_math`, `test_amd_matrix_fast` 2248 s, `test_amd_gcn_same_bits`, `test_amd_gcn_tier_same_bits` 2685 s) and TSan -O1 (`test_amd_mfma_math`, `test_amd_matrix_fast`) pass; the Release guards (`gcn_same_bits`, `gcn_packed_`, `gcn_memory_`, `codeobject_same_bits`, `nan_order`, `mfma_math`, tier digest) pass, and `--check-counters` shows no drift on the 37 pairs.

### Not done, and negative results
* A hint slot for two-way branches: no workload shows a hint miss (branch-misses of `divergent`: 1.7 M of 740 M branches).
* The per-instruction frame (counters, state flags): removing the four counter updates altogether gave 3 to 10% (valu_int 3%, smem 6%, divergent 10%), so batching them cannot give more; left.
* `flat_*`, `buffer_load/store`, `global_atomic_*` in the tier: `flat` costs the same per instruction as `mem` (48 ns) already, the cost is the memory work, not the frame.
* fp8/small-float decode by `vgatherdpd`: not tried, a gather costs about what the scalar table lookups do on Zen 4.
* AVX-512 for the tiles: Zen 4 runs it as two 256-bit halves, nothing to show here.

