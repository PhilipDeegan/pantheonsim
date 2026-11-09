## Round sp4: the SASS tier's run ends and diverged warps (branch `sp4-sassctl`, base 53721cb2)

Profiled first (A build, one host thread, `cycles:u`): `divergent` is 3.59 M runs of 2.8 warp instructions, about 100 cycles a warp instruction at IPC 3.5, branch misses 1% of
branches, no L1i misses: bound by instructions per run, not by stalls. `Runner::step_warp` (the tier's run loop inlined) was 50% of it, `pick_group_avx2` 13%, `blend_m_k` 8%.
`warp` is 1.28 M runs of 4 instructions, `step_warp` 33%, `shfl_move` 9%, `h_vote` 7.5%.

* **Known address** (`tier_run_t`, the chain): a run that ends by giving every lane of its group one address (a branch all the group takes or none, a handler-ended
  instruction such as SHFL or VOTE, BSSY, NOP) knows it; when no runnable lane stands outside the group the next group is the runnable lanes at that address and the 32 addresses
  are not read back. `warp` about 1.12x of its own.
* **VOTE** with a full warp writes the ballot as vector stores, not a loop over the lanes' bits: `warp` about 1.06x.
* **A branch's addresses for a partial group** (`sasslanes::set_pcs_partial`, `select_pcs`, `src/simd/lanes.cpp`): AVX2 load, blend, store of four lanes at a time, from a table of
  the sixteen nibble masks; the per-lane loops they replace had a trip count no predictor learns. `divergent` 1.12x. NOTE: the first version used `vpmaskmovq` stores and was
  slower than the loop (10% of `divergent` in the two functions, `divergent` 1.00x): masked stores are slow on Zen 4.

Result (B/A of medians, 11 alternating runs, cpu 0, counters identical, noise from A/A 0.98x to 1.04x): `warp` sass 1.184x, `divergent` sass 1.130x (both p < 0.001); every canary row
(sass and ptx) within the A/A noise except `wmma` sass 0.962x and `mma_ptx` sass 0.955x (15 runs, p 0.000 and 0.016). Those two run in `mma_avx2_chunked_f` (70% of the cycles, code this branch
does not touch) with 1% fewer instructions and 3 to 5% more cycles: a code-placement effect (the build without the blend piece has the same cycles as A (perf stat); a 16-byte frame pad in `step_warp`, whose
frame shrank by 16 bytes, and 64-byte alignment of the two new functions did not remove it). Not understood further.

Tests (CMake-wired): `test_sass_lanes` `set_pcs_and_select_pcs_agree_with_the_lane_loop`; `test_sass_block_reuse` (hand-encoded SASS, decode checked) `a_full_warp_vote_gives_every_lane_the_ballot` and
`a_diverged_warp_votes_among_the_lanes_that_run_it` (a split warp, a partial vote). Mutations, each caught: the known address `here + 32` (5 of 6 tests of `test_sass_block_reuse` fail), the vote fill
`ballot ^ (l == 5)`, `select_pcs` with target and next swapped, the nibble mask `& 7` in `set_pcs_avx2`.

### Not done, and negative results (this branch)

* *BSYNC's arrival mask worked out before the wait stores* (to avoid reading the wait arrays back from byte and word stores in flight): A/B against the tree with it, `divergent` 0.981x without it (p 0.23): not a win, dropped.
* *`shfl_move` Down and Up as SSE2 vectors* (unaligned loads inside the row, gathers at the edges): `warp` 1.017x (p 0.042, under 2%), dropped.
* *`vpmaskmovq` for the partial pc stores*: slower than the loop on Zen 4, see above.
## Round sp4: the SASS tier's memory handlers (branch `sp4-memhot`, base 53721cb2)

Profiled first (base build, one host thread, `cycles:u` and `instructions:u`, RTX 3060 profile): the memory workloads run at an IPC of 3.3 to 4.5, so they are bound by instructions, not by stalls.
`h_local` is 31% of `local` (a scalar address loop that reloads `w.local_size` and the frame vector's data pointer for every lane, because the stores to the register row may alias them), `h_global` + `tier_global16` 12% of `mem`, whose
address array (32 stores) and the second pass that compared it with a run were the cost of the coalesced case, which then never used the array.

### What changed
* **`h_local`**: the full-warp address loop works on 32-bit offsets (the same low 32 bits of base + uniform + immediate), the frame base and size are read once. `local` -7.4% instructions.
* **`h_global`, `tier_global16`**: a contiguous mask of lanes is tested for being consecutive elements on the registers' rows (`sassmem::global_run`, `include/vgpu/exec/sass_mem_rows.hpp`, SSE2, two lanes a step), and the 64-bit address array is built only
  for an access that is not a run or that the memory manager declines. `mem` -4.9% instructions, `shared` (its global loads) -3.6%.
* Two code-placement fixes found by the canaries, with equal instruction counts: `mma_avx2_chunked_f` (`wmma` read 0.973x as its start moved 32 bytes within a line) is aligned to 64 bytes; `tier_set_pcs` (`divergent` read 0.96x) too.

### Results (min of 7 runs of cycles and the instruction count, `perf stat`, one host thread; base 53721cb2 / this branch; the box ran at load 15 to 50, so the 9 to 15 run A/Bs below are noisy)
mem 576.1 M / 533.3 M cycles (0.926x), 2410.6 M / 2292.3 M instructions; shared 137.8 / 134.4 (0.976x), 637.3 / 614.6; local 465.8 / 410.5 (0.881x), 2193.9 / 2030.4; atomics, warp, divergent, wmma, mma_ptx, int_alu, f32_fma within 1% (cycles), instruction counts equal.
`bench.py --ab`, 11 alternating runs, final tree, cpu 4, load 20 to 30: mem 1.097x, local 1.112x, shared 1.024x, warp 1.031x (not changed here: placement), atomics 1.024x (not changed here: placement), divergent 1.004x, wmma 1.002x, all p < 0.001 where "faster"; PTX rows 0.99x to 1.02x (same); counters identical in every row. A/A: 0.996x to 1.004x on mem, shared, local.

### Tests
`test_sass_mem_rows` (CMake-wired): `global_run` against the lane loop over runs, runs with one lane a bit off, carries into the high half, lo-only rows, no row, element sizes 1 to 16, masks of any start and length. Mutants killed: the second lane's expected address off by one,
the high rows ignored, the last odd lane skipped, a null base counted as a run. `h_local` is covered by the bench `local` self-check (`bench_counters`): a load that reads the lane's frame 4 bytes off fails `thread 0`.

### Not done, and negative results
* *Bank-0 splat rows* (a per-block cache of the 32-lane row a constant-bank-0 word is spread into by `tier_dyn`, to avoid 128 bytes of 16-byte stores read back as 32-byte vectors): 0 instructions saved (2292.3 M with and without), cycles not separable from noise at load 30; adds a 2.6 KB member to `Block`. Dropped.
* *`h_shared` on the registers' rows* (the same idea for a warp's 32 consecutive shared words or one word every lane names): 1.5 M of 637 M instructions (0.24%) saved on `shared`. Dropped.
* *Atomics, a warp that names one word keeps it in a register through the lanes* (instead of a store-to-load chain through `cur[slot[l]]`): `atomics` 1.010x (15 runs) against 1.007x without it. Dropped.
* *Closed form for the sector count of a private array* (`sectors_fast` is 22% of `local` with `sector_window_k` and `distinct_sectors`): `on_stride` already turns a warp that is not on a stride away after four lanes; not changed.
* PTX engine: not changed (exec_ld/exec_st read, no piece measured); its `mem` row moved 0.91x in cycles with equal instructions, `int_alu` +1.6% cycles, `api` +1.5%: placement, not chased.
## Round sp4: the AMD engine's carry instructions (branch `sp4-amdprof`, base 53721cb2)

**Measured first** (perf `cycles:u`, one host thread, the bench-amd binaries of both targets, A = the round 3 build): `h_carry`, the tier's handler for `v_add_co_u32`, `v_addc_co_u32`, `v_sub_co_u32`, `v_subb_co_u32` and the reversed forms, was a scalar loop over the active lanes (a swap, a carry-in shift and a 64-bit sum a lane), twice. RDNA code builds every 64-bit address with that pair, so on rx6800 it took 35% of `mem`, 26% of `flat` and 19% of `narrow` (the same workloads cost about twice as much on rx6800 as on mi300x, whose compiler uses `v_lshl_add_u64`). Ranking of the other rows (share of the process, rx6800 / mi300x): `buf` `buffer_words` 36 / 44% plus the batch memory functions 25%; `atomics` `wave_atomics` 37 / 44%; `ldsx` `h_ds` 42 / 45%; `narrow` `wave_scalars` 17 / 28%; `smem`, `divergent`, `wave` the per-instruction frame (`run_groups` 23 to 40%) with no handler over 11%; `valu_*` handlers 10 to 21% of the process, the rest the bench binary's self-check.

### What changed
* `k_carry<SUB>` (`gcn_tier_kernels.inc`, table `kCarry`): the eight shapes as lane-row kernels, 8 lanes at a time under AVX2 (carry-in bits expanded to lane masks, unsigned compares by a sign-bit bias, the carry mask by `movemask`), the old per-lane arithmetic at the baseline level; all n lanes are computed and the handler masks the carry by EXEC and merges the sum (`h_carry_body`, `TIER_DUAL`, `Tables::carry`).

### Results (B/A of medians, one host thread, counters identical in every row, `--check-counters` matches)
| workload | mi300x | rx6800 |
| --- | ---: | ---: |
| `mem` | 1.01 | **1.51** (p < 0.001) |
| `flat` | 1.01 | **1.29** (p < 0.001) |
| `narrow` | 1.01 | **1.35** (p 0.03; 0.9 to 1.3 in other runs while the server was loaded) |
Canaries (`valu_*`, `mfma*`, `smfmac`, `wave`, `lds`, `divergent`, `launch`, `api`, `buf`): all within 0.97 to 1.03; single rows that read 0.955 (`smem` mi300x), 0.974 (`atomics` rx6800) and 0.978 (`mfma`) in one run read 1.31, 1.03 and 1.26 (p > 0.6) in the next with the code moved: noise on a shared box, not a change (no code of those paths changed).

### Tests and mutations
`test_amd_gcn_tier_same_bits` (and its `simd_base` twin) already runs every carry shape of the corpora with the tier off and on over the carries-and-borrows flavour and half the lanes off. Mutations (built and run with `VGPU_TIER_ONLY=_co_`): carry-in bits read from lane 0 for every group of 8 (`cin & 0xFF`), and the borrow as `x <= y`: both fail it.

### Buffer dwords and whole-wave scalars (same branch)
`buffer_words` (the whole-wave MUBUF dword path) and `wave_scalars` were instruction-bound (IPC 5): the per-lane loop re-read the resource limits through the `fits` lambda's references (stores to `lanes[]` and `index` forced reloads) and went through `word` / `set_word` for every lane. Now the limits are a small by-value `BufLimits` (`fits(at, bytes, index)` takes the lane's index), the operand rows are taken once (`row_of`), and the loops read and write the rows directly; the slow lane-by-lane loop uses the same `BufLimits::fits`. Host instructions of `buf` on rx6800: 11.76 G to 9.54 G, cycles 2.34 G to 1.89 G.

| workload | mi300x | rx6800 |
| --- | ---: | ---: |
| `buf` | **1.40** (p < 0.001) | **1.26** (p < 0.001) |
| `narrow` (this piece alone, `wave_scalars`) | **1.08** (p < 0.001) | 1.28 with the carry kernels (this piece's share not separable) |
Canaries `smem`, `lds`, `wave`, `divergent`, `valu_int`, `mem`, `flat`, `atomics`: 0.97 to 1.04, counters identical. Tests: `test_amd_gcn_memory_same_bits` (buffer corpora over valid, invalid, too small and swizzled resources, and the wave-wide paths against lane by lane) fails under: the out-of-range lanes not zeroed, the index read from the wrong lane, the high word of a 64-bit store taken from the low row, the high word of a 64-bit load written from the low. Not covered by it (the corpus never sets RDNA's OOB_SELECT 1 with an index at the record count): the `index <= records` boundary. The same-address memo of `wave_scalars` is unobservable (equal addresses read equal values), a mutation of it survives.
`atomics` (`wave_atomics`, 37 to 44% of the process) has no single hot spot (the samples are spread over `atomic_data`, `atomic_result` and the batch loop); left.

### The tensor float kernel's placement residual (branch `sp4-place`, on `sp4-integration`)

`wmma` and `mma_ptx` (SASS) read 0.96x of the base in the integration tree with 1% fewer instructions. `perf stat` showed the cause: the front end. `op_cache_hit_miss.op_cache_miss` 11.7 M -> 43 M and
`de_src_op_disp.decoder` 110 M -> 449 M ops, all inside `tc86::mma_avx2_chunked_f<8,false>`, whose code was unchanged (same instructions, 64-byte-aligned or not; a sweep of the body's offset by 8 to 56
bytes with `.nops` moved it by 2%, never back to the base). The function's eight unrolled rows each carried their own copy of the unlikely path (the scalar lanes: infinities, NaNs, out-of-range scales,
rounding), a third of the function, between the hot vector lines. That path is now `mma_f_fix`, out of line and cold. Op-cache misses 15.5 M, cycles on `wmma` 478 M against 490 M (base) and 515 M (before).
B/A vs the base (11 runs, cpu 0, counters identical): `wmma` sass 1.065x, ptx 1.044x, `mma_ptx` sass 1.036x, ptx 1.026x; `divergent` 1.12x, `warp` 1.15x, `local` 1.14x, `mem` 1.10x, `shared` 1.03x;
the other canaries 0.998x to 1.02x. Unit tests: `test_tc_ampere`, `test_tc_ampere_fast`, `test_fastpath` pass.

## Round sp5: the LDS wide accesses (branch `sp5-amdprof`, base d28fe2ba)

**Measured first** (`ldsx`, mi300x, one host thread): `h_ds` was 48% of the process; perf annotate put the samples in two scalar loops, the transposition of a `ds_write_b128` run into the lane-major LDS layout (`tmp[w*l+k] = in[l]`, one word at a time) and the per-lane `for k < w` loop of the general `ds_read_b96/b128`, which re-read the operand's register number from memory on every word. IPC was 4.2 (instruction-bound, 7.4 G instructions, 128 M L1 misses that this change does not touch).

### What changed
`gcn_tier.cpp`: `lds_to_rows` / `rows_to_lds` move four lanes at a time with a 4x4 transpose (two words a lane: unpack and shuffle) for 2 and 4 words, for the consecutive-lane runs and, with every lane on, for any addresses (the lanes stored in lane order, so overlapping words end as they do one at a time); a partly-off wave and the 3-word forms use `ds_read_lanes<W>` / `ds_write_lanes<W>`, cold noinline functions with the width known and the rows in registers.

### Results (`perf stat` user cycles, 11 alternating runs of A = d28fe2ba and B; the box was loaded (load 27 to 35), so the bench harness's p values were not usable; counters identical)
| workload | cycles A/B | instructions A to B | p |
| --- | ---: | ---: | ---: |
| `ldsx` mi300x | **1.22** | 7.41 G to 5.77 G | < 0.001 |
| `ldsx` rx6800 | **1.14** | 9.97 G to 8.34 G | < 0.001 |
| `lds`, `wave`, `smem` (mi300x), `wave` (rx6800) | 1.015, 1.000, 1.000, 0.998 | equal | > 0.4 |

### Tests and mutations
`test_amd_gcn_tier_same_bits` (and its `simd_base` twin; run with `VGPU_TIER_ONLY=ds_`): fails under a swapped pair of stores in `lds_to_rows<4>`, a swapped pair in `rows_to_lds<2>`, a reversed word order in `ds_write_lanes` (`ds_write_b96`), and a skipped last group of the all-lanes loop.
## Round sp5: the NVIDIA API calls once more (branch `sp5-apilite`, base d28fe2ba)

`api` is 15 calls a round at about 140 ns each; perf (user cycles, call graphs) put 25% of it in `cudaMemcpy` (three copies a round), 13% in `cudaFree`, 6% each in `cudaMalloc` and stream create/destroy, and the rest flat: clock reads 6%, `__tls_get_addr` 4.5%, mutexes 6%, `ApiCall` 4%.

* **Device-to-device `cudaMemcpy`** went through a `std::vector<uint8_t> tmp(count)` (a malloc, a zero-fill, a free per copy; 2.6% of `api` in the vector's constructor alone, and the libc `rep stos` was the top symbol). The buffer is now the call's frame for up to 16 KiB (`kD2dFrameBytes`) and an uninitialised allocation beyond that. The source is read into it in full before the destination is written, as before (overlapping copies keep `memmove` semantics); every byte is written by the read, so nothing needs clearing.
* **`sticky_error()`** (asked by stream/event create, query, synchronize, destroy: the calls that check the context is alive) took the API lock to ask an atomic. `State::initialized` is now an atomic set with release after `rt`; once true the fault is read without the lock. Before the machine exists the locked path runs as before.

Results (`perf stat`, user, 3 x 15 runs on cpu 6, the whole process including start-up; the server's load was 25 to 60, so wall-clock A/B spread was 5 to 30%): `api` instructions 982.5 M -> 930.0 M (-5.3%), cycles 409 M -> 389 M (-4.8%, three runs: 409/389, 410/391, 503/395 in a noisy first); `launch` instructions equal (+0.04%). Bench A/B (`bench.py --ab`, SASS): `api` 1.043x (11 runs, p < 0.001) and 1.079x (15 runs, p 0.056); PTX 1.05x and 1.04x (p 0.15 and 0.32, spread 12 to 15%); A/A of the same two builds: 1.014x and 1.009x. `launch`, `startup`, `mem`: same (one PTX `launch` row read 0.976x with p < 0.001 in the first run and 0.987x, p 0.32, in the second; its instructions are equal).

Test (CMake-wired): `test_runtime_d2d_scratch` (every size around the 16 KiB frame and beyond, a copy from untouched memory over stale data, overlapping copies forwards and backwards, a refused copy leaves the destination alone, a context fault seen on every thread and cleared, readers racing a thread that sets and clears it). Mutations that fail it: the copy made in 4 KiB pieces (overlap corrupts: `an_overlapping_device_to_device_copy_reads_the_whole_source_first`), the lock-free path answering success (`a_context_fault_is_seen_by_calls_on_every_thread_and_cleared_again` and the racing test).

### Not done, and negative results

* **AMD `occupancy()`'s five `std::string::rfind` prefix searches per launch** (3% of AMD `launch` self time) replaced by compares of the first bytes: `launch` mi300x 0.949x (p 0.33), rx6800 1.008x (p 0.85), `api`, `stream`, `memcpy`, `graph` all `same` with spreads of 20 to 76% under load 25 to 60. Dropped: not measurable here.
* **Reusing `kernel_resources`' answer within a launch**: the lockstep pre-pass and `validate` each call it (3.3% of NVIDIA `launch` for both, `compute_occupancy`'s per-thread memo is itself 2.4%); one call saved is about 1.6%, under the bar. A memo on `EntryFn` would need the answer to survive concurrent launches of one kernel.
* **`Runner::init_block`** (8% of `launch`, two `rep stos` of 3 KiB each for the warp and its registers), `Runner::Runner` + `build_bank0` (6% + 3%, a vector allocation per launch), `read_launch_env` (2.8%, already one pass): not changed; each is a few percent and the launch rows were too noisy to see them.
* **AMD `launch`**: the AQL packet and kernarg allocate/free per launch (`MemoryManager::free` 7.7%, the exclusive table lock and the quarantine map) cannot be reused without changing allocation addresses; `getenv("VGPU_THREADS")` per dispatch is 1.5%.
* **The clock reads of `cudaMemcpy`** (two per copy, 6% of `api`) feed the telemetry's busy time and timestamp: kept.
## Round sp5: the SASS tier's control bound, measured (branch `sp5-sassctl`, base d28fe2ba)

Profiled the base (one host thread, `cycles:u`, `perf stat -M PipelineL1/L2`): `divergent` runs at IPC 4.2 to 4.3, **retiring 69%**, front end 8.5% (bandwidth 3.2%, op-cache misses 11 M),
bad speculation 3%, back end 19%; `step_warp` 44%, `pick_group_avx2` 15%, `blend_m_k` 7%. The line profile (`-g1`) is flat: no source line over 5%. The software popcount (`simd.hpp`) is
5.6% of the samples, `lanes_at` in `pick_group` 8.5%, `sass_lanes.hpp:100` (BSYNC release's `wait_arg` compare) 3.4%. `ls_bad_status2.stli_other` is 46 M (4.6 per warp instruction):
store-to-load forwarding failures, 15% of them in the BSYNC wait state (byte and word stores read back as 16-byte vectors), the rest spread over the run loop's counters and stack.

### Not done, and negative results (this branch)

* *Hardware POPCNT for `popcount32` (flag read once, `popcnt` through inline asm, software count otherwise) and a mask-table `pick_group_avx2` (one table load for the broadcast, and, compare of each
  vector) and `pick_group_diverged` called directly from the chain when the run did not leave the whole group at one address (skips the converged test `lanes_at`)*: 87 M fewer instructions of 3443 M
  (-2.5%) on `divergent`, cycles 804 M -> 800 M (-0.5%, inside the 0.3% to 1% run-to-run spread), B/A `divergent` 1.004x (p 0.13), `warp` 1.045x (p 0.15), `int_alu` 1.009x (p 0.065), `local` 1.008x:
  not a win, dropped. The bound is not the instruction count: the core already retires 69% of its slots; what is left is the dependency and forwarding chains.
* *PTX `divergent`/`warp` (`VGPU_SASS=0`)*: IPC 4.4, `run_warp_until_yield` 38%, `step` 16%, `exec_micro` 19%; the `-g1` line attribution points into `std::__fill_a`/`_Construct` (20% of
  the samples) but the inline chain could not be resolved to a source statement with the available tools, so nothing was changed.
* Not tried (no data pointing at them): `blend_m_k` without the old-row load, vector `tier_wait`/`wake` for partial groups (an earlier arrival-mask reordering that removed one read-back of the wait
  arrays measured 0.98x to 1.02x in sp4).

### The PTX interpreter's diverging branch (branch `sp5-sassctl`, measured, not kept)

The `-g1` line profile of PTX `divergent` listed 20% in `std::__fill_a`, `_Construct` and `vector::size`; those are inlined container accessors of the hottest functions, not fills. A `-g -fno-omit-frame-pointer`
build and `perf annotate -l` resolved them:

* `exec_micro`: `micro_src` spreads an immediate across 32 lanes whenever the one-entry-per-slot cache holds another constant: 30% of `exec_micro`'s samples (2.5% of the run). A 16-entry hash cache per slot
  (`SplatCache`) removed the fills and measured `divergent` ptx 1.011x (p 0.17), `warp` ptx 0.935x, `local` ptx 0.946x: nothing to gain, layout losses; dropped.
* `step`, `exec_bra`'s diverging case: `Path fall{fall_pc, fallthrough}; fall.issued_at = ...; w.paths.push_back(fall);` builds the temporary with two 8-byte stores and the copy into the vector reads it
  back as one 16-byte load, which store forwarding cannot serve: 53% of `step`'s samples sit on that one instruction (about 3.5% of the run). Building the path in place
  (`emplace_back` and three field stores) gives `divergent` ptx 1.027x to 1.029x (p < 0.001, counters identical) but moves the hot interpreter code: `int_alu` ptx 0.970x, `local` ptx 0.982x
  (p <= 0.001; same instructions, L1i misses 2.9 M -> 3.6 M, branch misses 0.7 M -> 1.0 M, op-cache misses +14%); the same change as a cold `noinline` helper: `divergent` 1.017x, `int_alu` 0.989x,
  `local` 0.958x. A rebuild of the unmodified source reads 0.994x to 1.005x of the base (so these are the change's layout, not the build). Not kept: a 3% gain on divergent PTX for a 2% to 4% loss on straight-line PTX.
  The interpreter's hot code (`run_warp_until_yield`, `exec_micro`, `step`) is larger than the L1i and any edit near it is a layout lottery.

### A layout regression and its fix (same branch)
With the wide LDS code inside `gcn_tier.cpp` the flat and global handlers read 2 to 12% slower (flat mi300x 0.905x, rx6800 0.876x on a quiet box; `perf stat` on server3: equal instruction counts, more cycles, `find_ordinary_miss` and `run_allocation` suddenly in the profile): the larger unit moved the link-time inlining and the layout of code that does not touch the LDS. Moving the code into its own unit, `gcn_tier_ds.cpp`, compiled with `-fno-lto` (the functions are plain calls taking row pointers), returned them: user cycles A/B flat 1.02 (mi300x) and 1.04 (rx6800), mem 0.99 and 1.01, lds 1.00, with `ldsx` 1.17 and 1.13; the tier digest tests (both levels), `gcn_same_bits` and `gcn_memory_same_bits` pass, and the same mutations fail `test_amd_gcn_tier_same_bits`.

### Not done, and negative results (this branch)
* AMD atomics: giving each 4-byte integer operation its own lane function in `wave_atomics` cut host instructions by 9% (mi300x) and 7% (rx6800) but user cycles by only 1.7% (p 0.005) and -1.6% (rx6800 slower, p 0.001), and the flat rows moved 0.975 to 1.056 with the change in `gcn_exec.cpp`: dropped.
### Round sp5: the PTX interpreter's .local words, and what bounds SASS atomics (branch `sp5-memhot`, base d28fe2ba)

* **PTX `ld.local` / `st.local` of a full warp of 4-byte words** (`exec_ld`, `exec_st`; `local_words_inside`, `local_gather_words`, `local_scatter_words`): the alignment and frame test is two reductions over the lanes' addresses (it was a lane loop that also built an offsets array, then a `LaneSet`, then `write_loaded`), and the words go between the warp's array and the 32-bit register's row directly; a store reads the register row, not a widened copy. PTX `local`: 6237 M to 4882 M instructions (-22%), cycles 1201 M to 1041 M (min of 7), `bench.py --ab` (11 runs, cpu 2) 1.159x, p < 0.001, counters identical. Test: `a_warp_of_local_words_matches_the_lane_loop` in `test_ptx_paths` (fast path against `VGPU_FASTPATH=0`: memory, counters, errors; per-lane index, uniform index, past the frame, the last word and the first word past it, a misaligned lane and every lane misaligned, 8-, 20-, 32-, 64-lane blocks). Mutants: a store scatter with the lane index xor 1 fails it (memory differs at word 0); the frame bound off by 4 bytes aborts it (heap overflow, `free(): invalid next size`); dropping the alignment test is NOT caught (the unit test passes; why the misaligned lanes still raise the same error is not understood, the guard is read, not proven).
* **Code placement, not chased**: PTX `atomics` read 0.944x with equal instructions (min cycles 178 M to 197 M): `load_at` (memory.cpp, an `lto_priv` clone) went from 5% to 13% of the cycles; PTX `int_alu` 0.961x (+1% cycles). Both move with any change in the library; memory.cpp is not edited in this round.
* **SASS atomics, what bounds it** (instruction profile with line numbers, `atomics` workload): no single bound. The lane-address/dedup loop of `exec_mem` (exec_ops.inc 3736 to 3757) is about 12% of the instructions, the operand `fetch` about 4%, the lane loop about 8%, `pthread_mutex_lock` 5%, `batch_addresses_ok` 5%, `load_batch_in`/`store_batch_in` 8%, the rest is spread over the tier and the launch. A register chain for one-word warps measured nothing in round 4; a pre-test for "every lane names one address" would save at most the dedup loop's per-lane compare (est. 4% of the instructions); not done.
* **PTX `mem`**: `int_bin64_lanes`/`micro_int64` (the 64-bit address arithmetic, 10% + 7%) are plain lane loops of the ALU path; not changed.
## Round sp8: the memory manager's hot paths (branch `sp8-memmgr`, base a6e80555)

Profiles of the memory workloads (Zen 4 box, one thread, `cycles:u`) put the manager's own functions at 5 to 26% of a row: the PTX atomics spent 26% in `rmw_locate` and its callers, `batch_ok` was 10% of PTX `mem` and 6% of SASS `atomics`, `load_fields` 9 to 11% of `texture`. What bounded each, from `perf annotate`:

* The batch scan (`batch_addresses_ok`) made a min / max / or pass over the 32 addresses (about 11 instructions an address: two conditional moves for the extremes, the range tests) and then asked `HostMaps::may_overlap`, an out-of-line call, on every batch; `sassatom::batchable` and every scalar path called `may_contain` out of line per address.
* `rmw_batch_fn` called `rmw_load` and `rmw_store` (out of line) per lane with the access width a runtime value: each lane went through a `load_at` switch, the alignment test and a call.
* A miss in the thread's lookup memory (a kernel that reads one allocation and writes another alternates between two, so every second access missed) hashed the thread pointer a second time and scanned the eight ways from zero; the callers also kept an address-taken `base` local (a stack protector frame in `rmw_locate`).
* `load_at` and `store_at` were out-of-line clones (5% of PTX `mem`) carrying their byte-wise fallback.

What changed (`src/core/memory.cpp`, `include/vgpu/memory.hpp`):

* `batch_addresses_ok`: an `or` of the addresses and one `(address - base) >= limit` flag each; a host mapping is ruled out for the whole allocation by one range test (`host_map_free_span`: the allocation lies in the device windows and no host buffer was ever mapped inside one), and only otherwise is each address asked. Same answers: an address inside the allocation is a device address, so `may_contain` cannot say yes for it unless a host buffer sits in a device window.
* `HostMaps::may_contain` / `may_overlap` are `always_inline` (no lock is added anywhere: the host-map rule stands, the checks are the same relaxed loads).
* `load_at` / `store_at`: the aligned access inline, the byte-wise path a cold noinline function.
* `rmw_batch_fn`: the plain (not lock-free) loop runs with the width a constant (1, 2, 4, 8) and `rmw_load` / `rmw_store` inline.
* `find_ordinary_miss` takes the caller's lookup memory, tries the way that answered before the last one first (`LookupMemory::prev`), and returns the allocation and base in registers.
* `run_allocation` tests alignment with a mask (`unit` is a power of two there), not two divisions.

### Results (Zen 4 box, cpu 2, 1 thread; B/A, 11 alternating runs; A is the base commit built the same way)

| row | bench.py B/A (p) | perf stat cycles B/A | instructions B/A |
| --- | --- | --- | --- |
| PTX `atomics` | **1.36x** (< 0.001) | 0.73 | 0.81 |
| SASS `mem` | 1.045x (< 0.001) | 0.95 to 0.98 | 0.98 |
| PTX `mem` | 1.005x | 0.96 to 1.01 | 0.98 |
| SASS `atomics` | 1.02x | 0.97 to 0.99 | 0.99 |
| SASS `shared` | 1.02x | 0.98 | 1.00 |
| PTX `texture` | 1.02x | 0.97 to 0.98 | 0.98 |
| SASS `texture` | 1.02x | 0.98 | 0.98 |
| AMD `atomics` (mi300x / rx6800) | **1.13x / 1.08x** | | counters identical |
| AMD `buf` (mi300x / rx6800) | **1.07x / 1.06x** | | |
| AMD `flat` (mi300x / rx6800) | 1.03x / 1.06x | | |
| AMD `launch` mi300x | 1.05x | | |

The A/A control (the same build under two paths) moves `mem` by up to 4 to 5% in the median ratio and the other rows by 1 to 2%: only PTX `atomics`, the AMD `atomics`/`buf`/`flat` rows and the instruction counts are beyond it. Canaries (SASS and PTX `int_alu f32_fma f64_fma wmma mma_ptx divergent warp local launch api`; AMD `valu_int valu_f32 mfma wave divergent smem lds ldsx api` on mi300x and rx6800) are within 0.98 to 1.02 with equal instructions; one AMD `lds` rx6800 run read 0.93 and the repeat 0.99 (A/A 1.00). Counters are identical in every row.

Tests: `test_memhot_sp8` (new; `load_fields` against one `load_scalar` per field over real, uniform and untouched chunks and across chunk boundaries; the batch and window accepted or declined exactly at the allocation's last element, one past it, below it, misaligned; a host buffer mapped inside the allocation declining the batch that reaches it and not the one beside it; two allocations used alternately across frees of another and of either), plus the existing `test_memfast`, `test_memrmw`, `test_memrmw_lockfree`, `test_memalloc`, `test_memory`, `test_fastpath`, `test_sass_atomics`.

| mutation | result |
| --- | --- |
| scan limit `size - 1` short (`a->size - size`) | `test_memhot_sp8` fails |
| `host_map_free_span` always answers yes in the batch scan | `test_memhot_sp8` and `test_memfast` fail |
| a stale lookup-memory way answered without revalidation | `test_memhot_sp8` fails |
| the plain rmw loop does not store an unchanged result when `always_store` is set | `test_memrmw` fails |

### Not done, and negative results
* `load_fields` in one chunk (one chunk-table read for the record, one range test for the host maps): 0.8% more instructions on `texture`, cycles equal. The function is latency-bound: 24% of its samples sit on the texel load itself and 12% on the chunk-pointer load, both cache misses of a strided texture, not on the checks. Dropped.
* `LookupMemory` cache-line aligned with the cursors first: SASS `texture` **1.20x more cycles** at 0.98x the instructions (frontend stalls 9 M to 27 M, L1 misses equal). The hot lines of the lookup path collide with the texture's strided stream; the layout the allocator gives by luck is better. Dropped; note that two builds of the same source also differed on SASS `texture` (0.996 and 1.05 in cycles), so a layout of the hot globals (`g_lookup_slots`, the lock epoch, the thread's memory) is worth re-checking after any change near it.
* `MemoryManager::free` / `remember_freed` / `backing::release`: already reuse the freed node and the chunk array; no cost left to take in the allocation-per-launch path beyond the exclusive lock.
* `run_allocation`'s mask (two divisions less) is not separable from noise on its own; kept with the rest because it is the same theme and costs nothing.
## Round sp8: the AMD instruction classes no benchmark exercised (branch `sp8-amdprobe`, base a6e80555)

`bench_amd.hip` has no kernel for DPP, `ds_swizzle`/`ds_permute`, `v_permlane16`, packed 16-bit integers, VOPD, the double-precision functions
(`ldexp`, `frexp`, conversions, `min`/`max`) or scalar loads of more than four words, so the rounds before this one could not say what they cost.
This round adds them as a second program (`tools/bench-amd/kernels/bench_amd_probe.hip`, workloads `p_*`: own cache key, so the cached
`bench_amd` binaries stay valid; `bench.py` runs them on mi300x, rx6800, rx7900xtx and rx9070xt), profiles them, and gives the ones that ran in the
interpreter a tier handler. Each kernel is self-checking against a host reference bit for bit, and every one of them was also run on a **real
RX 6800 (gfx1030, server2)** with every result matching: so the references (DPP row and wave directions, the swizzle and permute address rules,
`v_permlane16`'s selector, the 16-bit wraps) are the hardware's, not the interpreter's own opinion. The binaries (gfx942, gfx1030, gfx1100, gfx1201) are built
on server2 with the harness's flags and keyed as the harness keys them (server3 has no hipcc).

### Where the time went (perf `cycles:u`, one host thread, A = a6e80555, tier on)

Every instruction of these classes was a tier "no handler" case (none was handed back after a handler started): the interpreter's name ladders and
per-lane loops ran them.

| workload | where | share |
| --- | --- | ---: |
| `p_dpp`, `p_dpp_scan` (mi300x) | `Machine::cross_lane_alu` (the shuffle into a 64-entry array) and `vector_alu` behind it | 55% |
| `p_perm` | `OpName::operator==` (the string compares of the `ds_*` ladder) 36%, `lds_access` 27%, `write_lane` 13% | 76% |
| `p_pk16` (mi300x) | `more_alu` (`v_pk_*_u16`, a lambda per lane, `packed_bits` per half) | 72% |
| `p_f64x` | `vector_alu` lambdas for `v_ldexp_f64` (libm), `v_min/max_f64`, the conversions, `frexp` | 36% |
| `p_vopd` (rx7900xtx) | `vector_alu` of both halves, the frame of `step_interp` | 55% |
| `p_smem_wide` | `MemoryManager::load_scalar`, one lookup per word of an `s_load_dwordx16` | 30% |
| `p_i64` | `v_ffbh_u32` (and, on RDNA, `v_clz_i32_u32`, `v_alignbit_b32` with a 16-bit count) | 13 to 25% |
| `p_lane`, `p_cmp`, `p_sdwa`, `p_sbuf`, `p_pkf16` (CDNA) | already tier handlers; the host reference of the program dominates | |

### What changed

* **DPP** (`h_int_dpp`, `classify_dpp`): the VOP1/VOP2 integer forms (`v_mov_b32_dpp`, `v_add_u32_dpp`, `v_and_b32_dpp` ...): the first source gathered
  as `Machine::dpp_shuffle` does (`dpp_from` is its `dpp_source`, row and bank masks, bound_ctrl, FI, DPP8; a control the interpreter refuses is not
  classified), the operation by the integer lane kernels, written for the lanes the shuffle writes. Float DPP forms and the VOP3 DPP forms stay the interpreter's.
* **LDS cross-lane**: `ds_swizzle_b32` (bit-mask, quad-perm and rotate patterns) and `ds_permute_b32` in `h_ds` beside `ds_bpermute_b32`.
* **`v_permlane16_b32`, `v_permlanex16_b32`** with FI and BOUND_CTRL (`h_permlane`).
* **Packed 16-bit integers** (`v_pk_add/sub/mul_lo/mad/max/min/lshlrev/lshrrev/ashrrev` in `_u16`, `_i16`, `_b16`; `h_pk16`): the two halves as rows through 20
  lane kernels (the clamped forms of add, sub and mad are their own), `op_sel` and `op_sel_hi` choosing the half each source gives, constants per half as
  `constant_half` has them. Also `v_ffbh_u32`, `v_ffbl_b32`, `v_ffbh_i32`, and `v_alignbit_b32`/`v_alignbyte_b32` on RDNA3 and RDNA4, whose shift count is a 16-bit operand.
* **VOPD** (`Machine::tier_vopd`, `classify_vopd`): each half is classified when it is finalized (`decode_one` now finalizes the halves before the pair), and the
  pair runs X's handler, then Y's with X's destination as it was where Y reads it (both read before either writes), exactly as the interpreter's `Enc::Vopd`
  case does; a half that hands back (a NaN where the rule is the interpreter's) hands back the pair with the registers as they were.
* **Double precision**: `v_ldexp_f64`, `v_frexp_mant_f64`, `v_frexp_exp_i32_f64`, `v_cvt_f32_f64`, `v_cvt_i32_f64`, `v_cvt_u32_f64`, `v_cvt_f64_f32` (the interpreter's own
  expressions as 64-bit lane kernels; `h_l64n` for the 32-bit results) and `v_min_f64`/`v_max_f64` (the float forms' rule: opposite zeros and NaN results go to the interpreter).
* **Scalar loads**: a `s_load_dwordx2` to `x16` wholly inside one allocation is one `MemoryManager::load_run` (it declines wherever the words one at a time could differ).
* **A bug the probes found**: RDNA2 decodes the VOP3 `v_ashrrev_i16`, `v_max_i16`, `v_min_i16` as `_e64`, which the `_e64` to `_e32` mapping of the 16-bit forms did not cover:
  the interpreter stopped with "decoded but not implemented" on any program using them (a real RX 6800 runs them). Fixed in `short_name_of`.

* **RDNA4's 64-bit scalar arithmetic**: `s_add_nc_u64`, `s_sub_nc_u64`, `s_mul_u64` as scalar handlers (every pointer the compiler computes on gfx12 goes through them).

### Results (B/A of medians, 9 alternating runs, one host thread, cpu 4 of server3, counters identical in every row; A = a6e80555)

| workload | mi300x | rx6800 | rx7900xtx | rx9070xt |
| --- | ---: | ---: | ---: | ---: |
| `p_dpp` (quad_perm, row_shl/shr/ror, mirror, half-mirror) | **2.18x** | **2.21x** | **2.19x** | **1.84x** |
| `p_dpp_arch` (wave_shl/rol/shr/ror, row_bcast; row_share, row_xmask) | **2.03x** | **2.18x** | | |
| `p_dpp_scan` (DPP folded into add, max, and) | **2.34x** | **2.32x** | | |
| `p_perm` (ds_swizzle, ds_permute, ds_bpermute) | **6.07x** | **6.01x** | **6.25x** | **6.06x** |
| `p_permlane` | | | **4.36x** | **4.29x** |
| `p_pk16` (packed 16-bit integers) | **10.2x** | (A cannot run it) | | |
| `p_vopd` | 0.995 (no pairs on CDNA3) | 1.065 (no pairs) | **11.7x** | **9.4x** |
| `p_i64` (64-bit integers, ffbh/clz, alignbit) | **1.55x** | **1.35x** | **2.19x** | **2.13x** |
| `p_f64x` (ldexp, frexp, conversions, min/max) | **2.39x** | **2.24x** | | |
| `p_smem_wide` (s_load_dwordx2/x8/x16) | **1.62x** | **1.64x** | **1.67x** | **1.45x** |
| `p_cmp`, `p_sdwa` (VOPD cndmask, alignbit on RDNA3/4) | 0.98, 1.01 | 1.06 (noise), 1.01 | 1.59x, 1.29x | 1.49x, 1.79x |
| controls `p_lane`, `p_pkf16`, `p_sbuf` (already handlers) | 1.015, 1.07, 1.00 | 1.00, 1.01, 0.99 | | |

(p < 0.001 for every "faster" row except where the ratio is within 2% of 1; the rx9070xt rows were taken before `s_add_nc_u64` was added.)

Canaries (the main program's workloads, B/A of medians, 9 to 21 alternating runs in windows where the box carried load 5 to 40 from the other agents: a row inside 3% is
noise; counters identical in every row):

| workload | mi300x | rx6800 |
| --- | ---: | ---: |
| `valu_int`, `valu_f32`, `valu_f64` | 0.99, 1.00, 1.02 | 0.99, 0.97 to 1.03, 1.04 (noise) |
| `mfma`, `mfma_bf16`, `mfma_i8`, `smfmac` | 1.02, 1.02, 1.03 to 1.04, 1.01 to 1.07 | |
| `mfma_f64`, `mfma_f64_4b` | **0.93 to 0.97**, 0.97 to 1.02 (a placement effect, below) | |
| `wave`, `lds`, `divergent` | 0.99, 1.00, 1.01 | 1.03, 1.02, 1.01 |
| `smem` (the scalar loads of x1 and x4 now one lookup too) | 1.05 | 1.02 |
| `buf` (its descriptor is an `s_load_dwordx4`) | **1.11** | **1.20** |
| `flat`, `launch`, `api` | 1.02, 0.99, 1.00 | 1.01, 1.00, 0.98 |

`flat` on mi300x against the pre-round-1 build (/tmp/base0, d8ec6128's parent): **1.980x** for this tree and 1.958x for a6e80555 (B/A 1.02, p 0.19 in the later run; 1.03 in the first): the 0.978x of the
earlier A/B does not reproduce; the three builds read the same within 2% once the box is quiet.

### Placement (the lesson of round 1, again)

The first full build read `mfma_f64` 0.944x (p 0.006) with the same instruction count, and `perf stat` (ASLR off, 8 alternating runs each) gave +3.8% cycles. Two causes,
both frames and not code: a 1 KB array in `tier_vopd` and a 64-byte one in `scalar_load` sat in the frame of `step()` (`tier_mach` is inlined there), which moved every frame
below it -- `step_interp`, `matrix_multiply_fast` and the stack buffers of the matrix tiles -- so the tile kernel's loads and its store to D alias in the low 12 bits where
they had not. Both are out of line now (`tier_vopd`, `scalar_load_run` are `noinline`): the frames of `run_groups`, `step_interp` and `matrix_multiply_fast` are the same sizes as in A (0x128, 0x6b8,
0xfe0) and the cost came down to +1.8% (best of 8: +2.5%). The rest is a 64-byte code line: `matrix_multiply_fast` starts at 0x...690 where A's was at 0x...860, so its hot
inner loop (+0x2b4c to +0x2b86, 58 bytes) straddles a line one boundary 0x24 bytes into the loop where A's did at 0x14; the loop is unchanged. Not fixed here (it moves with
every change of the library's layout and cannot be told from the other agents' pieces on this tree); the other matrix rows read 0.99 to 1.03.

### Not done, and negative results

* **Float DPP and the VOP3 DPP forms** (`v_add_f32_dpp`, gfx11's `v_add3_u32_e64_dpp`, `v_mad_co_u64_u32_e64_dpp`): the compiler emits them only where it fuses a `v_mov_b32_dpp` into
  an arithmetic instruction; the probes show them at 1/12 of the DPP instructions. Left to the interpreter.
* **True16 forms** (RDNA3/4 `v_mov_b16`, `v_cvt_f16_f32`, `v_pack_b32_f16`): their halves of registers (`.l`/`.h`, op_sel) are not modelled by any handler; `p_sdwa` on rx7900xtx is
  `v_mov_b16` for 100% of what the interpreter still runs there. A design question (the half-register write), not done here.
* **`s_atomic_*`**: no target of the harness has them (CDNA3 and RDNA2 to 4 have no scalar atomics), so there is nothing to measure; the interpreter's are untested by a program.
* **`s_buffer_load` on RDNA3 and RDNA4**: `p_sbuf` has no descriptor layout verified against hardware for them (gfx1100 and gfx1201 are skipped by the probe); on gfx9 and gfx10 the handler
  already existed (scalar memory is the Machine's `tier_smem`) and the probe reads 1.00x.
* **`s_load` as one lookup** was dropped in an earlier round for `x1` and `x4` (1.015x); for `x8` and `x16` it is 1.45 to 1.67x (`p_smem_wide`).
* `p_lane` (readlane, writelane, readfirstlane), `p_cmp`, `p_sdwa` and `p_pkf16` on CDNA3 were already tier handlers: the time of those workloads is the host reference in the program
  (a libgcc half-float conversion for `p_pkf16`), and no change is made for them.

### Tests (wired in CMake) and the mutations that fail them

* `test_amd_gcn_tier_same_bits` (every instruction of every corpus, tier off against tier on, 21 flavours of state, three seeds, plus its `VGPU_SIMD=base` twin) now also runs the
  forms `amd/tools/tier-corpus.py` writes for the new handlers: DPP and DPP8 (every control, masks, bound_ctrl, FI), `ds_swizzle` and `ds_permute` offsets, packed 16-bit integers with
  `op_sel`/`op_sel_hi`/clamp and every operand kind, VOPD pairs of every instruction pair (independent, Y reading X's result, constants and literals, high registers),
  `v_permlane16/x16`, the double-precision functions with modifiers, and `s_add_nc_u64`. New corpora: `isa_corpus_tier_gfx1030/1100/1201.txt` (only what the new handlers
  take; the CDNA files only gained lines, none was removed). 33 142 shapes, 786 564 runs each way, 0 differing (Release; AVX2 and base).
* `test_amd_gcn_probe_forms` (new): RDNA2's VOP3 `v_ashrrev_i16`, `v_max_i16`, `v_min_i16` run, tier on and off, in every state flavour.
* `test_amd_gcn_same_bits`: 16 806 of 16 809 recorded digests unchanged; three (the gfx1030 VOP3 16-bit instructions above) were the text of "decoded but not implemented" and are
  recorded from this tree (noted in the test), the instructions being checked against a real RX 6800 by `p_pk16`.
* `bench_amd_counters`: the 29 new workload/target pairs of the probe program, 66 pairs in all, identical with the tier on and off (VGPU_AMD_TIER=0), and the 37 old ones unchanged.

Mutations (a one-line change on server3, the restricted test named failed, reverted; all 15 failed their test):

| # | mutation | result |
| --- | --- | --- |
| M1 | DPP `row_shr` reads `row + ((in_row + n) & 15)` | tier digest (`row_shr`) fails |
| M2 | `ds_swizzle` xor mask taken from the or field | tier digest (`ds_swizzle`) fails |
| M3 | `ds_permute` address shifted by 3 instead of 2 | tier digest (`ds_permute`) fails |
| M4 | packed integers: the high result's `op_sel_hi` bit ignored | tier digest (`v_pk_add_u16`) fails |
| M5 | packed integers: the clamped kernels never taken | tier digest (`v_pk_sub_i16`) fails |
| M6 | VOPD: Y never sees X's destination as it was | tier digest (`v_dual`) fails |
| M7 | `v_permlanex16` reads its own row | tier digest (`v_permlanex16`) fails |
| M8 | `v_min_f64`: opposite zeros not handed back | tier digest (`v_min_f64`) fails |
| M9 | `v_cvt_u32_f64` of a negative: `x < 0` for `!(x > 0)` (a NaN) | tier digest (`v_cvt_u32_f64`) fails |
| M10 | `v_ldexp_f64`'s exponent masked to 16 bits | tier digest (`v_ldexp_f64`) fails |
| M11 | `v_ffbh_u32` of zero is 0, not -1 | tier digest (`v_ffbh_u32`) fails |
| M12 | `s_load_dwordx2..x16` words stored in reverse | `test_amd_gcn_same_bits` fails (160 instructions) |
| M13 | the `_e64` to `_e32` mapping of the three 16-bit integer instructions off | `test_amd_gcn_probe_forms` fails |
| M14 | DPP `bound_ctrl` ignored | tier digest (`bound_ctrl`) fails |
| M15 | `s_add_nc_u64` off by one | tier digest (`s_add_nc_u64`) fails |

Sanitizers: ASan+UBSan at -O0 (`ASAN_OPTIONS=detect_leaks=1`, `UBSAN_OPTIONS=halt_on_error=1`) of `test_amd_gcn_probe_forms` and of `test_amd_gcn_tier_same_bits` restricted (`VGPU_TIER_ONLY`) to the
new handlers' instructions (DPP, swizzle, permute, packed integers, permlane, min/max/ldexp/frexp/cvt, ffbh, `s_add_nc_u64`, a VOPD pair) pass; the whole corpus under ASan (three to five times the
2700 s of before) and a TSan build (no new threaded code) were not run here. Release: the guards (`gcn_same_bits`, `gcn_packed_same_bits`, `gcn_memory_same_bits`, `codeobject_same_bits`, `matrix_fast`,
`mfma_math`, `nan_order`, the other `test_amd_gcn_*`) and the whole tier digest, AVX2 and `VGPU_SIMD=base`, pass.


### Round sp8: the AVX-512 tile kernels are the default where the CPU has them

`kAvx512Default` (src/simd/simd.cpp) is now `true`: on a host with AVX-512 F/BW/DQ/VL and VNNI the tile kernels of the tensor-core routes (wgmma, tcgen05, the integer and f64 mma forms; `tile_f32`, `tile_f32c`, `tile_i32`, `tile_f64`) run as 512-bit code. `VGPU_SIMD=avx2` (config key `simd: avx2`) keeps the 256-bit kernels, `base` the baseline, `off` no kernels; `simd: avx512` is now an accepted config value. The lane kernels, `level()` and the JIT's ISA detection are unchanged.
Measured with `nvidia_tensor_bench` (ns per warp instruction, same binary, `VGPU_SIMD=avx512` over `avx2`, one pinned core, 5 repeats) and with the benchmark workloads (9 runs):
* Zen 4 (8840U, 512-bit operations as two 256-bit halves): tcgen05 s8 0.63x, wgmma s8 0.89x, wgmma f16/tf32 0.94 to 0.95x, the rest 0.97 to 1.03x.
* Zen 5 (9950X, full width): tcgen05 s8 0.59x, wgmma s8 0.78x, tcgen05 f32 accumulate 0.89 to 0.90x, wgmma f16/bf16/tf32/e4m3 0.92 to 0.94x.
* Sapphire Rapids (Xeon 8488C): geometric mean 0.916 over 43 forms; tcgen05 s8 0.46x, wgmma s8 0.68x, tcgen05 f32 0.76x, layout B / e4m3 0.81 to 0.86x, mma f64 0.76x; a few small forms 1.02 to 1.21x (noise at 5 repeats).
* int_alu, f32_fma, f64_fma, wmma, mma_ptx, mem, shared, local, divergent (SASS and PTX): 1.00x on Sapphire Rapids, within noise on the others.
Tests: `test_simd_dispatch` (which table each `VGPU_SIMD` setting gives on the host; run for the default and for `avx2`, `avx512`, `base`, `off`), and the tile-route tests (`test_mma_fast`, `test_mma_tiles`, `test_wgmma_fast`, `test_tcgen05_tiles`, `test_wmma_generic_fast`, `test_tile_mma`) now run three more times, with `VGPU_SIMD=avx2`, `avx512` and (for the ones that had it) `base`, so every level stays tested on a host that picks AVX-512 by default.


## Round sp9: S2R special-register reads (branch `sp9-s2r`, base 873dbc69)

`Runner::sreg` computed SR_TID.X/Y with two integer divides per lane, out of line, and `h_s2r` called it once per lane. Two changes (src/sass/exec.cpp, src/sass/tier_ops.inc):
* `Runner::build_tid_tab` fills a table of SR_TID.X/Y/Z for every thread of the block at launch, by counters (no divide; the lanes past the block's last thread continue the count, as the division gives it). `sreg` reads the table. The table is a plain member (3 x 1056 words, no initializer, so a launch does not zero it).
* `Runner::tier_sreg_fill` gives `h_s2r` a one-pass fill for SR_TID.X/Y/Z, LANEID, CTAID.X/Y/Z and NTID (a memcpy of the warp's table row for a full mask, the active lanes only otherwise); any other register takes the old per-lane loop. `CS2R`, `init_block` and the PTX interpreter (which already reads precomputed `tid_x/y/z`) are unchanged.

### Results (A = 873dbc69, B = branch; real programs from tools/bench/real, one host thread, cpu 0 of server3, outputs identical)
| program | A s | B s | A/B | runs |
|---|---:|---:|---:|---:|
| samples/transpose | 15.97 | 10.33 | 1.55x | 9 (final build; the build before `noinline`: 15.84 to 11.58, 1.37x) |
| rodinia/gaussian | 6.05 | 4.59 | 1.32x | 9 (final build; before `noinline`: 5.59 to 4.85, 1.15x) |
| polybench/LU | 70.40 | 44.16 | 1.59x | 3 (long program, not enough runs for a p) |
| polybench/FDTD-2D | 185.51 | 130.80 | 1.42x | 3 (same) |
A/A (A against A, 9 runs, quiet box): gaussian 1.01x, transpose 0.99x. An A/A taken while another build ran on the box gave transpose 0.93x and an A/B in that window gave 1.37x on a base of 18.2 s: only the quiet-box rows above are quoted. HeCBench babelstream and aligned-types exceed the 600 s timeout on both sides (no number); the corpus has no HeCBench transpose.
bench.py canaries (13 workloads, SASS and PTX, 9 runs, counters identical in all 26 rows): all within 0.98 to 1.02 except `shared` PTX 0.975x (p=0.001, reproduced with 21 runs at 0.974x; the A/A of the same rows was 0.995x). The PTX engine is not touched; `perf stat` showed equal instructions and cycles +2 to 4%, L1i misses +30% (small), and a different LTO inlining of `run_warp_until_yield`/`exec_ld`, so it is a placement effect of the relink. Marking the two new functions `[[gnu::noinline]]` restored it to 0.990x (21 runs, p=0.004, inside the 0.98 bound); this is layout, not a measured cause. (LU and FDTD-2D were measured on the build before `noinline`.) `mma_ptx` PTX moved 1.04x in the first link and 1.01x in the last, again layout.

### Tests
`test_sass_s2r_fill` (CMake, Release and ASan+UBSan -O0): hand-assembled SASS stores each thread's TID.X/Y/Z, LANEID, CTAID.X/Y/Z, NTID and three guarded S2Rs, checked against host formulas on the tier and with VGPU_FASTPATH=0, for 1D/2D/3D blocks, odd and partial-warp sizes and 1024 threads. Mutations that fail it: Z filled from the Y table (1 of 3 pass), `sreg` Y read at `tid ^ 1` (2 of 3), the partial-mask fill writing inactive lanes (0 of 3).
Guards: `ctest -R 'sass|s2r|tier'` (25 tests: the SASS unit tests and `test_sass_s2r_fill`) passes in Release; all 26 canary rows show identical counters. The e2e programs (`e2e_sass_tier`, `e2e_sass_archs`, `bench_counters_tier`, ...) skip on server3 (no real nvcc), so they are left to CI. The sanitizer run covered `test_sass_s2r_fill` only (ASan+UBSan -O0); no TSan (no threaded code).

## Round sp9: shared-memory bank passes by warp shape (branch `sp9-bank`, base 873dbc69)

Where the time went (perf `cycles:u`, one host thread, A = 873dbc69; shares of the program's samples): `memcount::sass_bank_passes` was 40% of Samples
convolutionSeparable, 34% sortingNetworks, 31% histogram, 19% mergeSort, 21% Rodinia lud, 14% dct8x8, 11% hotspot, with `h_shared` another 7 to 25%. The closed
forms (`bank_passes_progression`) already take every warp on one stride; what is left is warps that are on no stride. Dumping one call in 211 of the real programs
(2.5 M records) showed that the left-over warps are mostly few shapes seen over and over: convolutionSeparable's row stage is two rows of 16 words 160 words apart
(2 passes, 600 000 samples, 2 distinct shapes), dct8x8 four shapes (groups of 8 lanes on one word, a row of 8 words four times), mergeSort / sortingNetworks pairs of
words every four and bitonic index patterns (about half of them repeat), segmentationTreeThrust 8-byte rows; only histogram's byte gathers are random.

### What changed

* `memcount::BankMemo` / `sass_bank_passes_memo` (include/vgpu/exec/mem_counters.hpp): the passes of a warp depend on which of its lanes' words are the same and which banks they are in,
  and moving every address by a whole number of words rotates the banks together and keeps equal words equal. So after the closed forms give up, the warp's shape (the width, and each
  lane's address less the first active lane's word; a lane that does not run is taken at that word, which adds none) is looked up in a 32-entry table, one per host thread
  (`count_shared_access` in src/sass/exec_ops.inc, a `thread_local`, so no lock). A hit compares 256 bytes of offsets; a miss counts as before and stores. Addresses of 2^62 and over go to the counting (a
  move could wrap them). When fewer than a third of the last 64 lookups hit (a gather, histogram) the table is left alone for 1024 calls, so a kernel with no repeating shapes only pays
  for a 64-lookup probe in 1088 calls. All the shared-memory counting sites of the SASS executor use it (`h_shared`, `exec_mem`'s LDS/STS, the generic-address path, LDGSTS).
* `sass_bank_passes_table`: the counting behind the closed forms, for 1 to 4 byte accesses, walks the active lanes with a counted loop (the run of consecutive lanes, as the sector code does) and the table's counts are bytes. Same sets
  of words, same answer (a 0-byte access keeps its old meaning: a word only at an unaligned address).
* `lane_progression<true>` (bank passes only) tests the last lane before walking the lanes, so a warp of rows is turned away at once (it was 16 lane pairs of work for convolutionSeparable's shape). The sector counting
  calls `lane_progression<false>`: its warps nearly always are on the stride, and the extra comparison cost `mem` / `local` 3.5% (SASS) in the first version.
* The PTX interpreter's `ptx_bank_passes` shares `BankTable` (byte counts) and `lane_progression`; it has no table of shapes.

### Results (Zen 4 box, `taskset` cpu 2, one thread, A = 873dbc69 built the same way, `real.py` whole programs, median wall, 9 alternating runs (5 for the programs of 20 s and more), Mann-Whitney p)

| program | A s | B s | A/B | p | A/A control (same builds) |
|---|---:|---:|---:|---:|---|
| Samples convolutionSeparable | 25.13 | 17.39 | 1.445x | 0.008 | 1.008x (p 0.42) |
| Rodinia lud (1024) | 4.63 | 3.86 | 1.199x | 0.0004 | 1.000x |
| Samples sortingNetworks | 21.93 | 20.25 | 1.083x | 0.008 | |
| Rodinia hotspot | 1.68 | 1.62 | 1.038x | 0.0004 | 0.998x |
| Samples histogram | 28.03 | 27.06 | 1.036x | 0.008 | |
| HeCBench shmembench | 12.04 | 11.72 | 1.027x | 0.0004 | 0.993x |
| Samples mergeSort | 10.25 | 10.20 | 1.005x | 0.03 | 1.003x |
| Samples segmentationTreeThrust | 31.36 (user 21.9) | 33.07 (user 21.0) | wall 0.948x, user 1.046x | 0.84 | not run (the wall time carries host work and a launch of the Thrust build) |
| Samples dct8x8 | 2.02 | 1.89 | 1.066x | 0.25 | 0.894x (p 0.43): under 2.2 s of which most is host code, not measurable |

`perf stat` of convolutionSeparable on a quiet box: 115.6 G cycles -> 79.7 G (-31%), 526.6 G -> 368.4 G instructions. Counters: `VGPU_COUNTERS=1` of all of convolutionSeparable, sortingNetworks, histogram,
mergeSort, dct8x8, segmentationTreeThrust and shmembench, A against B, every launch's block of counters is byte-identical (139 M, 100 M, 81 M, 15 M, 0, 160 M and 504 M bank conflicts).
The `bench.py` canaries (sass and ptx, 15 alternating runs, equal instruction counters): int_alu, f32_fma, f64_fma, wmma, mma_ptx, mem, shared, atomics, local, divergent, warp, launch, api all 0.986x to 1.029x (nothing under 0.98;
A/A of mem, local, wmma, mma_ptx sass 0.999 to 1.002x).

### Placement (the lesson of round 1, again)

The first version kept the table in the `Block` (8.7 KB at the end of the struct). `wmma` and `mma_ptx` (SASS) lost 3.5 to 5% with equal instruction counts and no extra misses: the
tile kernel `tc86::mma_avx2_chunked_f<8,false>` (a 10 KB function with a 4 KB frame) went from 64.8% to 70.7% of the samples. Moving the table out of `Block` (a `thread_local`, which leaves `sizeof(Block)` as it was)
brought both back (0.995x and 0.993x cycles, B faster). Aligning `step_warp` to 64 bytes, a `section` attribute for the new functions (a section type conflict with the comdat for an inline function) and `noinline` on the
counting wrapper changed nothing or made it worse: the kernel's speed follows the heap layout, not the code layout. Any later change to the size of `Block` or `Warp` should re-run `wmma` and `mma_ptx`.

### Not done, and negative results

* A branch-free "passes by levels" count (each level takes one word of every bank that still has words; the number of levels is the answer), in 32- and 64-bit words: 0.84x to 1.1x of the byte-count table on
  the sampled warps (convolutionSeparable 77 vs 64 ns, sortingNetworks 49 vs 50, histogram 95 vs 98, mergeSort 88 vs 88); not kept. A pre-test that every lane is in a bank of its own (the bank mask, one pass if it is full): 87 to 99 ns against
  60 to 80, slower, because a walk of 32 lanes costs about 80 cycles by itself (the tzcnt chain). A table of 8-byte keys instead of 32: no change.
* mergeSort and sortingNetworks repeat only some of their shapes (data-dependent masks and gathers do not; about half of the sampled sortingNetworks warps had a shape seen before in the sample, which was not measured as a hit rate in the run), so their gain (0.5 to 8%) is small; histogram (bytes, random words) only sees the leaner
  table. A memo in the PTX interpreter was not written (the programs here run on SASS).
* `h_shared` itself (15 to 25% of these programs after the change: address generation and the lane-by-lane data move) was not touched.

### Tests (wired in CMake) and the mutations that fail them

`test_bank_memo` (new, `vgpu_unit_test`): every answer of `sass_bank_passes_memo` against a from-scratch definition (sorted vectors of the words of each bank) with ONE table kept across 400 000 calls of
400 shapes (tile rows, groups on one word, pairs of words, windows, byte gathers, permutations, bitonic indices) at every alignment, widths 1 to 64 bytes, full / partial / sparse masks, idle lanes holding zero or anything,
rank bits in the high bits, the edge of 2^62 and the top of the address space; the same shape in every position of the banks and byte of the word; shapes that differ in one lane; that a shape is found again wherever it sits and
whatever the idle lanes hold (`hits`); that stride shapes never reach the table; the back-off; and `sass_bank_passes_table` / `sass_bank_passes` against `sass_bank_passes_reference` (the earlier counting, which `test_memfast` holds against the
original code) for widths 0 to 64 over 300 000 random warps. `test_memfast`, `test_memcount_forms` (also `VGPU_SIMD=base`) and `test_sass_access_count` pass unchanged.

| mutation | result |
|---|---|
| key ignores the width | `test_bank_memo` fails |
| the last two lanes left out of the comparison | fails |
| base not rounded down to a word (byte offsets shared) | fails |
| no 2^62 / top-of-space check | fails |
| the second word of an unaligned 4-byte access not counted | fails |
| 0-byte accesses counted like 1 to 4 bytes | fails |
| idle lanes taken at address 0 (or not replaced) | fails (a shape is not found again) |
| last-lane test off by one (nothing reaches the closed forms) | fails (stride shapes reach the table) |
| (key ignores the lane mask) | survives, by design: an idle lane is taken at the first word, which adds no word, so the mask cannot change the passes; the mask was dropped from the key |

Sanitizers on server3: `test_bank_memo`, `test_memcount_forms`, `test_sass_access_count` under ASan+UBSan at -O0 (leaks on, halt on error) pass; `test_bank_memo` under TSan -O1 passes (it is single-threaded; the table is a `thread_local`, no run with
threads was made beyond the programs above at one thread). The e2e guards that need `nvcc` (`e2e_sass_tier`, `e2e_sass_archs`) cannot run on server3 (stub compiler); the whole-program counter comparison above stands in for them.

## Round sp10: control dispatch of the tier (branch `sp10-dispatch`, base 9c1d6ffd = sp9-combined)

Where the time went (perf `cycles:u`, one host thread, A = 9c1d6ffd; shares of the program's samples): HeCBench bh 54% `step_warp`/`exec_control`/`fetch`/`code_at`/`Code::instr`, black-scholes
28% and bilateral 33% in `step_warp` + `exec_control`, and in the shared-memory programs `h_shared` 16 to 27%. `VGPU_TIER_STATS=1` showed why: of the instructions that end a tier run, the
interpreter was running `CALL.REL.NOINC` and `RET.REL.NODEC` (nvcc's float division slow path: black-scholes 17 M of 1.2 G warp instructions, bilateral 1.2 M of 84 M) and `FCHK` (always
"yes", see exec_float), and in bh the spin loop of its tree build (a block barrier, a load, a YIELD, then `CALL` and `WARPSYNC` round again, 546 000 turns at 600 bodies) ran `WARPSYNC` through
`exec_control`, whose `release_warpsync` called `Runner::fetch` (a section lookup and a decode check) once per released lane.

### What changed

* Tier forms `Fchk`, `Call`, `Ret`, `Warpsync` (src/sass/tier.hpp, tier.cpp, tier_ops.inc). `FCHK` is a straight-line form with a handler (the predicate set on the executing lanes). `CALL.REL` to a fixed
  address, `RET` and `WARPSYNC` end a run and are done by `Runner::tier_ctl`, a function of its own (the same counters as `execute()`: instruction, thread instructions, class; the lanes that run a call
  or return get the new address, the others the next instruction; `WARPSYNC` makes the lanes wait, then calls the interpreter's `release_warpsync`). A call whose target is one of the module's builtins
  (vprintf, malloc, ...) is left to `exec_control`. A uniform call or return keeps the warp's addresses as one address, so the chain goes on without a call of the scheduler.
* `Runner::release_warpsync` (src/sass/exec_ops.inc) is now a member (it was a lambda of `exec_control`) used by both engines. It looks at the waiting lanes only, takes one test when no lanes are
  held back and all waiting lanes name one mask (then they are all released or none is), looks the instruction at an address up once instead of once per lane, and wakes the lanes with
  `sasslanes::wake`. `WARPSYNC` reads a mask that is not a register (an immediate: `__syncwarp()`) once, not once per lane.
* `h_shared` (whole warp): the 32 offsets are worked out as 32-bit words in plain loops (`base << log2(stride) + add`, the sum of the low halves, which is the low half of the sum) and the window
  check is one reduction over them (`SharedWindowCheck::add_warp`, include/vgpu/exec/window_check.hpp), not 32 dependent `add()` calls (`xor` after `imul` carried 45% of the function's samples).

### Results (A = 9c1d6ffd, B = branch; `real.py`, one host thread, cpu 4 of server3, outputs identical; the box load was 20 to 40 for the whole round: only large effects are quoted)
| program | A s | B s | A/B | runs |
|---|---:|---:|---:|---|
| HeCBench bh (600 bodies, 1 step; the default 5000 x 2 runs for more than 300 s) | 109.5 / 106.5 / 98.2 | 70.7 / 71.1 / 64.3 | 1.55x | 3 pairs (an earlier build: 1.39x, then 1.63x with the WARPSYNC release) |
| HeCBench bilateral | 2.65 | 1.91 | 1.39x | 5 (A/A of the same box: 1.02x); an earlier build 2.31 to 1.91 (7 runs) |
| HeCBench black-scholes | 37.80 | 35.16 | 1.08x | 3 (an earlier build 31.52 to 28.32, 1.11x, 3 runs) |
| HeCBench shmembench | 17.25 | 14.71 | 1.17x | 5 (A/A 0.95x: noise of this program here is 5%) |
| Rodinia hotspot | 2.50 | 2.24 | 1.12x | 5 (A/A 0.99x) |
| Samples convolutionSeparable | `perf stat` instructions 364.8 G to 333.6 G (-8.5%), cycles 79.8 / 82.7 G to 75.1 / 75.1 G | | 1.06 to 1.10x | 2 pairs (wall time of the run: 26.97 to 27.26 s, 0.99x: the loaded box hides it; instructions are the reliable number) |
| Rodinia lud, Samples sortingNetworks, histogram | 1.00x, 0.98x, 1.00x | | | no change expected (lud and sortingNetworks keep their time in bank counting and non-full warps; histogram in atomics) |

bench.py canaries (12 workloads, SASS and PTX, 15 runs, counters identical in all 24 rows): all within noise (p > 0.05) except `mma_ptx` PTX 1.025x. One real placement effect on the way: the first
version (the three branches written into the run loop `tier_run_t`) cost `atomics` SASS 0.956x (p=0.001, 0.960x with 21 runs; A/A 1.001x) at equal instructions; moving them to `tier_ctl`, a `noinline`
function, gave 0.992x (15 runs, p=0.05) and 0.995x (21 runs, p=0.35). The load of the box for the canary runs was 20 to 40 (spreads of 5 to 20%): rows between 0.96 and 1.0 with p above 0.1 (`mem` SASS 0.958x,
`wmma` PTX 0.968x, `api` PTX 0.969x) could not be told from noise here and are not explained by anything the change touches (the PTX engine is not touched).

### Tests (wired in CMake) and the mutations that fail them
* `test_sass_dispatch` (new): hand-assembled sm_86 kernels. A loop calls a function whose return address is in a register pair (every lane; the upper half warp only, so the callee and the code
  after the call both run for different lanes; a RET some lanes do not run, which then fall into an EXIT), with `FCHK`, and `WARPSYNC` with the full mask and with a register mask of the lane's half warp
  (also after lanes have exited); eight shapes of whole-warp shared-memory access (in order, every other word, scrambled, bytes, one address, `.X4`, `.X8`, 64-bit) in warps of 32, 64, 33 and 5 threads. Each
  run is compared tier against the interpreter (`VGPU_FASTPATH=0`) in registers and every counter, **and** against a digest recorded from the tree before the tier ran these instructions (the
  interpreter and the tier share `release_warpsync` and the window check, so comparing them with each other cannot show a mistake in what they share; `VGPU_PRINT_DIGESTS=1` prints the table, taken from the
  9c1d6ffd build with `-DSASS_CALL_RET_NO_TIER_FORMS`). A last test requires the analysis to take all four new forms.
* `test_tier`: the form table and the corpus rules know the four forms. `test_memcount_forms`: `add_warp` against the per-lane window test (400 000 random warps).
* Mutations that fail them (each applied alone, test rebuilt): CALL target + 16; RET target + 16; FCHK false; WARPSYNC mask ignored (every lane names the full mask); RET counting one thread instruction;
  release without the "every named lane has arrived" test; the one-mask path ignoring held-back lanes; the release counting exited lanes as missing; shift of the shared offsets one too many;
  the offset's add term dropped; `add_warp` keeping the last maximum / last other-window bit only. All fail `test_sass_dispatch` or `test_memcount_forms` (1 of 6, 1 of 7 or 1 of 9 tests, with 4 failing checks for the first three).
  One mutation was not caught at first (the release counting exited lanes) and led to the test "WARPSYNC after exits"; a section lookup shortcut in the release was not covered by any sm_86 test (a `.COLLECTIVE` WARPSYNC exists from sm_90) and was removed.
* The e2e programs (`e2e_sass_tier`, `e2e_sass_archs`, ...) skip on server3 (no real nvcc) and are left to CI.

### Not done, and negative results
* bh's spin loop is the program's own algorithm (a block barrier inside a wait loop); the remaining time (600 bodies, perf `cycles:u`) is 45% `step_warp` (the tier loop), 9% `tier_ctl`, 7% `release_warpsync` (it was 15% in the first version of the release), 5% `tier_release_bar`.
* `BRA.CONV` / `BRA.DIV` (6.9 K and 3.9 K of 590 K warp instructions of bh at 500 bodies) and `P2R` stay with the interpreter; not worth a form.
* convolutionSeparable keeps `h_shared` 19% (the per-lane copies), `load_scalar` 7% (one call per `FFMA` that reads a `__constant__` word through `c[0x3][..]`: not cached, because the memory manager's fault
  injection must see every load) and the bank-pass table 10%; sortingNetworks / lud are the same shape. Nothing was changed there.
* Not run: TSan (no threaded or atomic code was touched), the AVX-512 versus AVX2 tensor-core re-check (TensorCoreGemm samples; the box was not quiet enough for a 5% question).

## Round sp10: atomics, zero-copy accesses and directed rounding (branch `sp10-atomics`, base 9c1d6ffd = `sp9-combined`)

Where the time went (perf `cycles:u`, one host thread, A = 9c1d6ffd; shares of the program's samples): `Runner::exec_mem` was 23% of HeCBench aop (binomial options, shared-memory atomics), 29% of
Samples histogram with its clones, 22% of radixsort2, 16% of atomicCost (global `RED` of float, double and int), and 24% of zerocopy plus `lookup_host`/`scalar_location`/`load_scalar`/`store_scalar`
(a kernel whose buffers are `cudaHostAlloc`-mapped: every lane's access went through the scalar loop); `fesetround` + `fegetround` were 23% of Samples interval (directed-rounding `DADD`/`DMUL`/`DFMA`).
`perf annotate` of aop's `exec_mem` put 20% of its samples on one instruction after a `__memcpy_chk` call: the shared-atomics fast path read each lane's word with a `memcpy` of a run-time size into a zeroed
`uint64_t` (a 4-byte copy into a word zeroed by an 8-byte store, then an 8-byte load: a store-forwarding stall per lane) and called `apply` out of line for every lane.

### What changed

* `sassatom::shared_batch` (include/vgpu/exec/sass_atomics.hpp): a warp's `ATOMS` on the block's shared bytes in a loop with the word width (4 or 8) and the step (add, f32 add, f64 add, min, max, signed min/max, inc, dec, and, or, xor, exchange,
  compare-and-swap) known when it is compiled, one out-of-line function per step, operands fetched for all lanes first and the old values written afterwards. `ATOMS.POPC.INC` (an `atomicAdd` of the constant 1) takes the add loop. Operations
  without a step (the f16x2/bf16x2 halves, min/max of the float types, an operation that is none) take the lane-by-lane code as before, so its faults are unchanged.
* New steps (`AddF64Step`, `MinS32/MaxS32/MinS64/MaxS64Step`, `IncStep`, `DecStep`) for `step_for`: the global batch's `lanes(apply)` general path had run all of these, `atomicAdd(double)` among them (atomicCost's third).
* `sassatom::global_batch`: the global batch's lane loop, a function per step with `__restrict` operands, and a one-word case (every lane on one address: a counter, a maximum) that keeps the word in a register from lane to lane
  instead of loading and storing it through `cur[slot[l]]`. (The first version inlined all sixteen steps into `exec_mem`; GCC then stopped inlining them and the bench `atomics` row fell to 0.91x: see below.)
* The global batch's address grouping (`exec_mem`, RED and ATOMG): three shapes are told apart in one pass over `ga[]` (no stores) before the 64-entry hash table is touched: all lanes on one address (the table is skipped, and the batch
  needs no `slot`), addresses rising from lane to lane (the lanes' addresses are the distinct ones, in order, no table), anything else (the table, as before). `ATOM` (generic addresses, classified lane by lane) keeps its old single loop.
* `MemoryManager::host_load_batch` / `host_store_batch` (src/core/memory.cpp) and their use in `exec_mem`'s LDG/STG batch: when the lanes' accesses all lie in ONE host-mapped buffer (one `lookup_host` for the warp) each is the relaxed atomic
  access `load_scalar`/`store_scalar` make on a host mapping (no device fault applies there); an access outside the buffer, misaligned, a store to a read-only mapping, or two buffers in a warp declines with nothing done and the scalar loop reports it. No lock
  (the host-map rule). A store resets `idle_yields` as `mem_write` does.
* `RoundingScope` (the directed-rounding scope around the SIMD kernels of FFMA/FADD/FMUL, DADD/DMUL/DFMA, I2F, F2F and the conversions) reads and writes the rounding field of MXCSR (bits 13 and 14, which are SASS's 0 RN, 1 RM, 2 RP, 3 RZ)
  directly on x86-64 instead of `fegetround` + `fesetround`, which also reload the x87 control word and cost four times as much (99 ns against 26 ns for a get, a set and a restore, measured). The sticky flags are untouched. Other hosts keep the libc calls.
  The scalar `rounded()` helper keeps `fesetround` (its arithmetic is inline, and the compiler is only kept from moving it across an opaque call).

### Results (Zen 4 box, server3, `taskset` cpu 2, one thread, A = 9c1d6ffd built the same way; outputs identical)

`real.py`, median wall of 9 alternating runs (5 for the programs over 30 s), and `perf stat` of one run each (instructions are deterministic; cycles are the better time under the load of a shared box):

| program | wall A s | wall B s | A/B wall | instructions A/B | cycles A/B |
|---|---:|---:|---:|---:|---:|
| HeCBench zerocopy | 90.99 | 44.45 | 2.05x | 2.35x | 2.19x |
| Samples histogram | 39.41 | 31.46 | 1.25x | 1.38x | 1.36x |
| HeCBench atomicPerf | 2.21 | 1.71 | 1.29x | 1.33x | 1.35x |
| HeCBench radixsort2 | 4.58 | 3.95 | 1.16x | 1.19x | 1.17x |
| Samples interval | 27.54 | 23.60 | 1.17x | 1.02x | 1.16x |
| HeCBench aop | 3.43 | 3.09 | 1.11x | 1.22x | 1.16x |
| HeCBench atomicCost | 44.10 | 36.63 | 1.20x | 1.15x | 1.14x |
| HeCBench atomicAggregate | 7.83 | 7.19 | 1.09x | 1.01x | 1.03x |

A/A (A against A, 9 runs, the same window, the box at load 12 to 30 from another session): aop 0.96x, atomicPerf 0.94x, radixsort2 1.04x, interval 0.99x. Wall ratios under about 1.06x are inside that noise; the cycle and instruction columns are the claim for those rows
(atomicAggregate is match/popc-bound, its atomics are one word: no claim). Not claimed: lavaMD, particlefilter, binomial (the fp64 `dop_avx2_t` is already an 8-lane AVX2 kernel; see below).
`bench.py` canaries (13 workloads, sass and ptx, 15 alternating runs, counters identical in all 26 rows): every row `same` (no verdict slower; `mem` sass 1.000x, `atomics` sass 1.024x, `shared` sass 1.010x, `warp` 0.990x, `int_alu` ptx 0.910x median with best-of ratio 0.985x and p 0.34,
an engine this round does not touch). The first version of the global batch showed `atomics` sass 0.907x (p 0.001, equal counters): instructions +2.5%, cycles +5% in `perf stat`, from the sixteen inlined step loops (GCC then called them out of line through a lambda) and a word kept in memory from lane to lane; the one-word register loop and the per-step functions
bring it to +0.6% instructions and 1.02x in the harness.

### Tests
* `test_sass_atomics` (unit, CMake): every new step against the frozen rule; `shared_batch` against the lane-by-lane `memcpy` + frozen rule over offsets that repeat, step by 4 and overlap by half a word, full and partial masks, all operations and types, and the declines leave the bytes alone; `global_batch` against the frozen rule over shared slots, the single-slot loop included.
* `test_hostmap_cache` (unit, CMake): `host_load_batch`/`host_store_batch` against `load_scalar`/`store_scalar` lane by lane at every width, consecutive, scattered and repeated addresses (the later lane wins); the declines (outside, misaligned, below the base, a width that is none, two buffers, read-only) change nothing; and the host's own relaxed atomics race the batch (clean under TSan).
* `sass_fastpath.cu` (e2e `e2e_sass_fastpath`, fast paths against `VGPU_FASTPATH=0`: answers and every counter): `atomic_more` on global and on shared memory over the four address patterns (signed 64-bit min/max, constant-one add with its old value, i.e. `ATOMS.POPC.INC`, `RED` of the same, signed 32-bit max, f64 add) with a host reference for the final memory; `hostmap_forms`: 1, 2, 4 and 8 byte loads and stores on `cudaHostAlloc` mapped buffers, whole and partial warps, identity, strided, reversed and broadcast lanes, host-checked. The program was built on server2 (CUDA 13.0) and run on server3 against both builds: A, B, and B with `VGPU_FASTPATH=0` print the same 4,247 lines (`wrong: 0`).
* Mutation evidence (each fails the test named): `MinS32Step` taking an unsigned compare, the single-slot loop without the width mask, `shared_lanes` taking the lanes in another order, `IncStep` with `>`, `DecStep` with `old - 2`, `AddF64Step` with the NaN operands tested in the other order, `changed` always set, `shared_batch` not storing: `test_sass_atomics` 2/5 to 4/5 pass; `host_store_batch` accepting a read-only mapping, no alignment check, no bounds check, stores in reverse: `test_hostmap_cache` 6/7 pass. Wiring, one build with five mutations of `exec_mem`/`RoundingScope`: the POPC.INC add of 2 (`atomic_more space=1`, WRONG in all four patterns), the rising path giving every lane slot 0 (`atomic_forms` and `atomic_more` pattern 2), `host_store_batch` skipping the last lane (`hostmap_forms`, WRONG in all four patterns), `RoundingScope` setting the next mode (`float_forms`, `half_forms`, `conv_forms`: 15 hash lines differ), shared CAS given no exchange value (`shared_atomic_forms`, WRONG in all four patterns).
* Sanitizers on server3: `test_sass_atomics` and `test_hostmap_cache` under ASan + UBSan (-O0, `detect_leaks=1`, `halt_on_error=1`): clean. TSan -O1 (`setarch -R`, g++ 13 with `VGPU_WERROR=OFF`: it rejects `atomic_thread_fence` under `-Werror=tsan`) `test_hostmap_cache` (the batch racing a host thread's relaxed atomics included) and `test_sass_atomics`: no report. `e2e_host_atomics` under TSan was not run here (needs a TSan build of the shim and nvcc from server2); the new host-mapped batch takes no lock and uses the same relaxed atomics as `load_scalar`/`store_scalar`, so it is left to CI.
* Guards: Release `ctest -R 'sass|tier|mem|hostmap|atomic|lanes|cache|simd'` (the 107 matching tests that were built) passes except `memory_totals`, which needs the `vgpu` CLI that was not built; the e2e programs need a real nvcc and run in CI; `sass_fastpath` was run by hand as above.

### Not done, and negative results
* fp64 `dop_avx2_t`/`dop_m_k` (lavaMD, particlefilter, binomial; 25 to 27% of those programs): already an 8-lane AVX2 kernel doing the arithmetic and one NaN test per 8 lanes; what is left is the arithmetic. A 512-bit form would run as two 256-bit halves on this Zen 4 box, so it cannot be measured here.
* Batching `fesetround` across consecutive instructions (interval alternates RM and RP, so the mode changes every instruction; only the restore could be saved, and every other piece of host arithmetic assumes round-to-nearest). Replaced by the cheaper MXCSR read-modify-write above.
* `RoundingScope`-style MXCSR access for the scalar `rounded()` helper: its arithmetic is inline, and an `ldmxcsr` intrinsic is not an optimisation barrier for it as the opaque call is.
* A host-mapped run form (`load_run`/`store_run` for consecutive lanes): `host_load_batch` already takes the whole warp with one lookup; zerocopy's remaining time is the simulator's own 3 loads and a store per element.

## Round sp10: the global gather (branch `sp10-gather`, base 9c1d6ffd = `sp9-combined`)

A warp whose lanes are not on consecutive words (a column walk: PolyBench SYRK/SYR2K/2MM/GRAMSCHM/3MM, the cuRAND state loads of the Samples MC_*) goes through `h_global`, `MemoryManager::load_batch`, `batch_addresses_ok`, `load_batch_in` and `memcount::distinct_sectors`: 50 to 60% of SYRK before this round (perf, one thread). Three changes, none of which moves a result or a counter:
* `MemoryManager::load_batch` (src/core/memory.cpp) makes the tests of `batch_addresses_ok` (aligned, inside the allocation, by one comparison with `size - width + 1`) in the same pass as the loads: per lane the chunk is looked up and the word read, `gather_loads`; a lane in an untouched or uniform chunk takes the old per-lane code; a lane past the end reads the allocation's first byte, whose result is dropped when the batch declines. Batches of more than 32 accesses and allocations that a host mapping may be in take the old two steps. `store_batch`, `batch_window`/`batch_load` (the atomics' pair) are unchanged. `load_batch`'s `out` is now of no use after a decline (it used to be left as it was; every caller reads it only after a success): a copy through a buffer, to keep the old guarantee, cost 4% of SYRK, so `test_memfast` no longer asks for it.
* `h_global` (src/sass/tier_ops.inc) handles a full warp (`ex == kAll`) without the lane compaction: the addresses are the batch as they stand, the constant-window test is made in the pass that builds them (`sassmem::full_warp_addresses`, include/vgpu/exec/sass_mem_rows.hpp: the smallest distance above the window, below 2^32 means a lane is in it; the separate pass over the stored addresses waited for the stores), and the loaded words go back to the registers by row. Partial warps keep the compaction.
* `memcount::distinct_sectors_uncapped` (include/vgpu/exec/mem_counters.hpp), the SASS executor's `count_access` entry: the closed forms in a small function, the list fallback (an 8 KB array, so a stack frame that probes two pages on entry, 30% of `distinct_sectors` for a strided gather) in a `noinline` function of its own. The PTX interpreter keeps the original `distinct_sectors`.

### Results (A = 9c1d6ffd, B = branch; cpu 0 of server3, one host thread, `perf stat` user cycles, 9 alternating runs; outputs and counters identical)
| program | A cycles (G) | B cycles (G) | B/A | instructions B/A | p |
|---|---:|---:|---:|---:|---:|
| polybench/SYRK | 51.65 | 41.25 | 0.799 (1.25x faster) | 0.815 | <0.001 |
| polybench/SYR2K | 103.94 | 81.53 | 0.784 (1.27x) | 0.809 | <0.001 |
| polybench/3MM | 16.04 | 14.43 | 0.900 (1.11x) | 0.875 | 0.005 |
| samples/MC_SingleAsianOptionP | 11.08 | 10.22 | 0.922 (1.08x) | 0.921 | <0.001 |
| samples/MC_EstimatePiInlineP | 9.19 | 8.41 | 0.915 (1.09x) | 0.911 | <0.001 |
A/A (A against A, same script): SYRK 1.001, 3MM 1.000. 2MM and GRAMSCHM run past 150 s on the box and were profiled only (the same functions, 50 to 60% of their time); CORR and COVAR are 99% the program's own host code (CORR 0.993x, p=0.57). `bench.py` `mem` SASS 1.04x (p=0.3, not a claim). A server3 window with the other session's builds running gave the same ordering with more spread; the quoted rows are from the quiet window.
Where SYRK's time went (`perf record`, one thread): before, `h_global` 25% + `load_batch_in` 24% + `batch_addresses_ok` 9% + `load_batch` 3% = 61%; after the gather change, `load_batch` 31% + `h_global` 25% (the first is now the cache and TLB misses of the loads), `sectors_fast` + `distinct_sectors` + `count_access` 12%. The window fusion and the `distinct_sectors` prologue were taken after that profile.
bench.py canaries (13 workloads, SASS and PTX; 24 rows, counters identical in all): the A/A of the same rows spreads from 0.976x to 1.027x. B against A is inside that spread for every SASS row (`mma_ptx` SASS 1.037x p=0.004 is a gain, not a goal) and for PTX except `launch` PTX 0.960x (17 runs, p<0.001; `perf stat`: equal instructions, cycles +4.4%, branch misses +25%) and `local` PTX 0.97x (15 runs of `perf stat`: cycles +2.1%; bench.py gave 0.915x with p=0.02 in a loaded window). Neither runs any code this round touches; the functions of the PTX interpreter (`run_warp_until_yield`, `setup_block`, `launch_grid`) have identical sizes in A and B and sit at addresses that differ by 0x20 to 0x30 modulo 64, so this is the relink's placement again (the sp9 lesson), not a measured cause. `wmma` PTX: 0.99x here, and 0.962x in the build that had split `distinct_sectors` for both engines.

### Not done, and negative results
* Two passes (chunk pointers first, then all the loads in a tight loop, with and without a prefetch): 1.128x and 1.134x cycles on SYRK with 1.5% fewer instructions; the stalls move to the loads, and the single pass is 0.875x of the original in the same window (modes of one binary, quiet box). A column walk at 4 KB stride puts the 32 lanes in one L1 set and in 32 pages, so what is left of `load_batch` (31% of SYRK) is the cache and TLB misses of the loads themselves.
* Copying the words through a local buffer so that a declined `load_batch` leaves `out` alone: SYRK 0.838x and SYR2K 0.835x instead of 0.80x and 0.79x.
* Splitting `distinct_sectors` itself (as the PTX interpreter also uses it): the SASS gains were the same, and PTX `wmma` went to 0.962x (p<0.001, equal instructions; the hot `mma_avx2_chunked_f` moved from 0x164100 to 0x165d40, a placement effect). The split is now only the SASS entry; `wmma` PTX 0.99x, `mma_ptx` PTX 1.01x on the final build.
* Not started: an AVX2 gather for 4-byte lanes (the loads are miss-bound, not instruction-bound), and a sector count fused into `full_warp_addresses` (`sectors_fast` is 4 to 5% of SYRK).

### Tests (wired in CMake) and the mutations that fail them
* `test_gather_sp10` (new): `load_batch` against `load_scalar` per lane over real, uniform (two bytes) and untouched chunks, widths 1/2/4/8, 1 to 32 lanes and more than a warp, lanes spread over the allocation, one lane past the end, below the base or misaligned (the batch declines), and the window form. Mutations: uniform chunk read as zero (fails: "lane 4 of 9 at width 8 differs"), the bounds test dropped (fails: `VCHECK(!done)`), the alignment test dropped (fails the same way).
* `test_sass_mem_rows`: `full_warp_addresses` against the lane loop and the window test, with lanes on the window's first and last byte and one either side. Mutations: distance off by one (fails at `nearest < kSpan`), the distance biased for high addresses (fails the same way).
* `test_memcount_forms`: `distinct_sectors_uncapped` against the definition, on the SASS rows of its random warps (it passes through the closed forms and the list fallback). Mutation: the closed-form count `n` returned as `n + (n > 3)` fails 7 of 9 tests ("sass sectors (uncapped) 65 vs 64").
Guards run and passing (Release): `test_memory test_memfast test_memhot_sp8 test_memalloc test_memrmw test_memrmw_lockfree test_memcount_forms test_ptx_mem_runs test_ptx_paths test_ptx_block_reuse test_sass_*` (atomics, lanes, mem_rows, access_count, code_at, block_reuse, alu_runs, lea, mma_tiles, decode, ok_cache), `test_tier`, `test_amd_gcn_same_bits`, `_memory`, `_packed`, `test_amd_mfma_math`, `test_amd_matrix_fast`, `test_amd_nan_order`, `test_amd_codeobject_same_bits`; bench.py counters identical in all 24 canary rows. The e2e programs skip on server3 (no real nvcc) and are left to CI.

## Round sp5: process start (branch `sp5-startup`, base a6e80555)

Measured on server3 (cpu 2 alone, `perf stat -e instructions:u,cycles:u,task-clock`, 25 alternating runs of A and B per row, medians; the box was never quiet, load 12 to 60, so the
instruction counts are the figure to trust and the cycle and time ratios are given as measured). A = a build of the base tree, B = this branch.

* **Where a small program's start went.** `perf record` over 60 runs of the `startup` workload (SASS): 47% of user cycles in `ld.so`, and a single loop of it, the GNU symbol hash
  (`h = h * 33 + c`), was 25% of the process. The loader hashes the name of every symbol it looks up, and `libcudart` (and `libcuda`, `libcublas`, ... every shim that links the core) exported the
  `std::variant` dispatch tables the PTX parser's and the SASS decoder's `std::visit`s instantiate: weak symbols whose mangled names are 1,000 to 2,000 characters long (317 relocations, 669 of the
  754 thousand name characters the library's relocations carry; `readelf -r`). Nothing outside the library refers to them. `vgpu_version_script` now lists `_ZNSt8__detail9__variant*` as `local:`
  (GNU ld takes the more specific `local:` pattern over `global: *`; checked on a small library), and `libvgpucuda`, which had no version script, gets an unversioned one with the same line.
  Only these tables are hidden: the rest of the core's symbols stay exported on purpose, because the shims share one copy of the core's singletons by symbol interposition
  (`-Bsymbolic` or `-fvisibility=hidden` would give each shim its own). Relocation name characters in `libcudart`: 754,000 -> 85,000; `libcuda`: 733,000 -> 70,000.
* **`sass::unsupported` leaves its verdict in a file** (`src/sass/ok_cache.{hpp,cpp}`). The answer is a function of the cubin's bytes and of this build's decoder and executor, so a process that finds a
  cubin runs in full appends the 128-bit hash of the image (a four-lane multiply-rotate hash, one pass over the bytes) to `$XDG_CACHE_HOME/pantheonsim/sass-ok-v1-<GNU build id of the library>` (else
  `$HOME/.cache/...`), and a later process of the same build answers "runs in full" from it without parsing or decoding (a PyTorch sweep starts a dozen processes that load the same fatbins). Only passes are
  recorded, so a refusal keeps its text; it was opt-in (`VGPU_SASS_CACHE=1`) when it shipped (#438) and is on by default since round sp11 (see below); `VGPU_SASS_REFUSE` and `VGPU_SASS_CACHE=0` bypass the file; the file is used only if it is a regular file of this user's that no one else can write, in a directory of
  the same kind, and a torn tail is read around and never appended to; a build with no GNU build id has no file; at most 65,536 keys. A word table (`OkWords`) still serves the cubins that miss, but a process
  that answered from the file did not fill it for them.
* Results (SASS `startup`, `perf stat`, B/A of medians over 25 runs; instructions retired, user cycles, task clock):

| row | instructions | cycles | task-clock |
|---|---:|---:|---:|
| `startup` sass, variant tables hidden only (the cache off) | 0.69 | 0.80 | 0.93 |
| `startup` sass, file warm (second and later process) | 0.34 | 0.58 to 0.60 | 0.80 |
| `startup` ptx (tables hidden; the file is not used) | 0.95 | 0.975 | 0.95 |
| libcuda + libcudart preloaded into `/bin/true` (load only) | 0.27 | 0.36 | 0.74 |
| A/A control of the `startup` sass row | 1.000 | 0.998 | 0.984 |

  `bench.py --ab` `startup` wall (15 runs): sass 0.006 -> 0.005 s (1.29x, p < 0.001, the A/A run that followed read 0.003 s for the same build on both sides: wall at 3 to 6 ms is noise-bound), ptx 1.04x (p = 0.004); every
  other canary row (`api`, `int_alu`, `f32_fma`, `f64_fma`, `wmma`, `mma_ptx`, `mem`, `shared`, `atomics`, `local`, `divergent`, `warp`, `launch`, both engines) within 2.3% with counters identical (A/A
  of the same window: 0.95x to 1.02x, p > 0.1 on every row; `mma_ptx` read 1.04 to 1.05x, p <= 0.002, and `wmma` sass 1.016x p 0.03: the library's code layout moved, no claim). The PyTorch sweep (torch 2.10, `VGPU_THREADS=1`, 8 alternating rounds, instructions retired, B/B control 1.0000) shows what
  the file is worth: the symbol change alone moves a group by 0.05% to 0.25%, but a warm file takes 12% (`sdpa`), 28% (`nn`) and 36% (`ops`) off the instructions (minimum wall 10%, 13% and 27% lower, results identical), and a process that finds the file empty pays 0.6% to 2.2% more. One group leaves 21 to 88 entries (336 to 1,408 bytes).
* Tests: `test_sass_ok_cache` (new, CMake-wired): an image that ran is answered from the file and any other (one more word, another architecture byte) is a miss; a refusal is never recorded and keeps its text;
  `VGPU_SASS_REFUSE` and `VGPU_SASS_CACHE=0` bypass the file (a recorded image is refused when the instruction it holds is named); a torn tail is read around and not appended to; a file or directory others can write
  is not trusted and not appended to; the key covers every bit and the length. `test_unsupported_memo` sets `VGPU_SASS_CACHE=0` (it watches the word table, which a warm file would bypass). `lint_shim_dynsyms` (new, `tests/lint/check_shim_dynsyms.sh`): no shared library of the build
  exports or relocates a `std::variant` dispatch table symbol (it fails on the base build's `libcudart` and `libcuda`: 317 of each).

### Not done, and negative results
* AMD startup (`startup` mi300x, 4.4 M cycles, perf over 60 runs): `ld.so` 33%, libc 26% (malloc), `libamdhip64` 21%. `libamdhip64` already has its version script, so there is no long-name hash cost to remove; what is left
  is spread out: the decode tables are built at first use as `std::map`s (`add_comparisons`, `_Rb_tree` inserts, and `table_gfx90a` copies the whole map again, together about 5% by symbol, more with their `malloc`), the code object's msgpack parse
  (1.7%). A constexpr sorted array for the decode tables is the larger refactor this points at; not done here.
* Hiding more than the variant tables (`-fvisibility-inlines-hidden`, `-Bsymbolic-functions`): the remaining 860 relocations against `vgpu::` names (67 thousand characters) are the core's own functions called across
  translation units; the shims share singletons through them, so they stay exported.
* The remaining `ld.so` time of the SASS `startup` row (55% of what is left) is the loader's relocation and mapping work for libstdc++, libc and the shim, and the lazy binding of the PLT.

### Round sp11: warpfix

After the round 10 merge the `warp` workload (SASS) ran 0.91x of main at equal instruction counts. The kernel is 100% tier (no CALL, RET or
WARPSYNC), and the profile put the extra cycles in `step_warp` (the run loop) and `shfl_move`, which is unchanged code. The cause was the
signature of `tier_ctl` (the noinline CALL/RET/WARPSYNC function of sp10-dispatch): it took the run loop's `at` and `all_at` by reference,
so the compiler kept both in memory for the whole loop. `tier_ctl` now returns them (`CtlOut`, two registers). `perf stat` of `warp`
(cycles:u, 1 thread): main 241M, merged round 10 261M, fixed 243M. bh, zerocopy: cycles equal to the merged tree within 1% (bh 506/509 G vs
509/504 G, zerocopy 113.2 vs 112.2 G); bh runs 0.6% fewer instructions.

## Round sp10: large device-to-device copies chunk to chunk (branch `sp10-new`, base 9c1d6ffd)

Profiled first, on server3 (one host thread, `perf record -e cycles:u`, one pass over 120 programs of tools/bench/real, each stopped at 25 s, base 9c1d6ffd): outside the SASS/PTX engine, the tier handlers and the SIMD kernels there is nothing left that is more than a few percent of a real program. The memory manager's lookups (`load_run`, `run_allocation`, `find_ordinary_miss`, `store_run`: 2.4%, 2.3%, 1.5%, 1.3% of PolyBench on average) belong to the gather path; `sass_bank_passes_*`, `tex_footprint_linear`, `load_fields` and `exec_mem` belong to other rounds. What is host-side and not the engine is process start (a 3 ms program: `ld.so` 25%, libc 30%, the simulator library 37%, spread over some 40 symbols, none above 1.5%; `devrt::names()` and `yamlish::parse` are 1% and 0.6%) and the program's own CPU reference check (PolyBench `*_cpu`, `lud_verify`: 20 to 50% of those programs, which no simulator change reaches). The wall-minus-user gap of `fpdc` (273 s wall, 7 s user) is the program writing a 512 MB file through a throttled disk (`rq_qos_wait`), not the simulator.

A microbenchmark of the host-side API (a C program against the shim, `cudaMemcpy` of 64 MB in each direction, memset, malloc/free, 4 B copies) found one path far from the host's own speed: **device-to-device copies of more than 16 KiB** ran at 1.3 GB/s where H2D and D2H run at 8.6 GB/s. `cudaMemcpy` (D2D), `cudaMemcpyPeer`, `cuMemcpyDtoD` and HIP's `copy_region` read the whole source into a buffer (`new uint8_t[count]`, or a zero-filled `std::vector`) and write it out: two passes over the data plus the page faults of a fresh 64 MB mapping every call.

`MemoryManager::copy_to(dst_mm, dst, src, len)` (src/core/memory.cpp) copies chunk to chunk with `memcpy` and no buffer. It does what `read` then `write` leave, chunk for chunk: a source chunk that is untouched or uniform is a repeated byte (from a per-thread 64 KiB fill), and a destination chunk that already holds the bytes it is given as one repeated byte stays unmaterialized, an untouched one is made real, exactly as `write()` does (`resident_bytes()` is compared in the test). It returns false having done nothing, and the caller makes the old pair of calls and meets any error where it did, for: a fault armed on copies (or loads, stores, stuck cells), a host mapping in reach of either end of either range, a range not wholly inside one ordinary allocation (freed, mapped by cuMemMap, exported, out of range, `external`), and two ranges of one allocation that overlap. Callers: runtime_api.cpp (`cudaMemcpy` D2D above the 16 KiB frame, `cudaMemcpyPeer`), driver_api.cpp (`cuMemcpyDtoD`, not for managed memory), amd/src/hip_api.cpp (a D2D row above 16 KiB; the row buffer is allocated only when a row falls back). The function is `[[gnu::cold, gnu::noinline]]`.

### Results (server3, one thread, cpu 5; A = 9c1d6ffd, B = branch; alternating, 15 runs; counters not involved)
| op (64 MB unless noted) | A ms | B ms | speed-up | p |
|---|---:|---:|---:|---:|
| `cudaMemcpy` D2D, aligned | 40.7 | 6.3 | 6.45x | <0.0001 |
| `cudaMemcpy` D2D, unaligned ends (+4097 / +100) | 41.0 | 7.2 | 5.7x | <0.0001 |
| `cudaMemcpyPeer`, device 0 to 0 | 43.9 | 7.3 | 6.0x | <0.0001 |
| `cudaMemcpy` D2D into a fresh `cudaMalloc` (first touch) | 68.3 | 35.2 | 1.94x | <0.0001 |
| `cudaMemcpy` D2D 8 MB x 8 | 4.4 | 1.7 | 2.5x | <0.0001 |
| `cudaMemcpy` D2D 1 MB x 50 | 2.4 | 1.2 | 2.0x | <0.0001 |
| `hipMemcpy` D2D (mi300x) | 67.0 | 6.9 | 9.7x | (2 runs each) |
| H2D, D2H, pinned, 2D pitched, memset, 4 KB D2D (unchanged paths) | | | 0.97 to 1.02x | A/A moves them 0.98 to 1.02 |
`perf stat` of a loop of 300,000 4-byte D2H, 1 MB malloc/free, 4-byte H2D: instructions B/A 1.0000 (cycles 1.012, 1.006, 0.956 on a loaded box). A 4 KiB D2D loop read 1.8x in cycles once with identical instructions and 1.0x in a repeat (a neighbour on the box).

bench.py canaries (13 workloads, SASS and PTX, 15 runs, quick size, load 12 to 15; counters identical in all 26 rows): every row within 0.97x to 1.034x of A; the A/A of the PTX rows read 0.997x to 1.006x. Three PTX rows sit below 0.99x and were reproduced in two more windows: `local` 0.982x (p<0.001; 0.97 to 0.98x elsewhere), `api` 0.970x and `launch` 0.978x (p 0.1; 0.987 to 0.99x elsewhere). The instructions are equal; the added function moved the library's code layout (a first build without `cold`/`noinline` read PTX `local` 0.94 to 0.97x, `launch` 0.97x; `cold` took the launch row back). The SASS engine, the default, is level (`mem` 1.026x p 0.007, `launch` 1.018x). Not attributed further: any change that adds code to the library moves the PTX interpreter's layout by this much (see round sp9).

### Tests
* `test_memcopy_direct` (new, CMake): against the `read` + `write` pair on two identical worlds, 1,800 random copies (two allocations of one manager, two managers, disjoint halves of one allocation) over chunks that are untouched, uniform, real with random bytes, real with one repeated byte and half-written, at ends near chunk boundaries: bytes and `resident_bytes()` of both sides equal; the declines (overlap in both directions, past either end, a foreign manager, a freed range on either side, a host mapping where a range starts or ends, a fault armed on copies) return false and leave both buffers as they were; untouched memory copied over written memory gives zeros and makes the chunks real as `write()` does. Mutations that make it fail: skipping the uniform-chunk check (`disjoint_halves`), dropping the overlap decline, reading an untouched source as 0xFF (4 tests), dropping the copy-fault decline (the decline test).
* `test_runtime_d2d_scratch`: a new case of 400,000 bytes at 65,537 / 130,000 offsets through `cudaMemcpy` and `cudaMemcpyPeer`, the bytes around the range unchanged (the existing 300,000-byte, overlap and past-the-end cases now take the new path or the decline).
* `test_amd_hip_d2d_direct` (new, CMake): `hipMemcpy` D2D of four large ranges at odd offsets, forward and backward overlap as a memmove, a 2D copy of 100,000-byte rows, and a copy past the end refused with the destination untouched.


