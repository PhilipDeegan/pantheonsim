## Round sp14 (address rows and run loop, PR sp14-ship): gather-bound memory ops and the SASS run loop
Two changes in one round, measured separately on b4a51d7f (the sections below) and together on main a28b4972 (the last section). Part 1 is the address-row kernel for full-warp global accesses; part 2 is the run loop that no longer ends a call at the register-only ops and a convergent BRA.
### Part 1: a full warp's global address rows in one AVX2 pass (branch `sp14-addr`)
Profile of the gather-bound PolyBench programs on main (`perf record` cycles:u, one thread): `h_global<false>` is 17-25% of ATAX, GEMM and SYR2K outside `load_batch`, and in ATAX's annotation about half of it is the loop that builds the 32 addresses from the registers' lo/hi rows and takes the running minimum of `address - window` (the compare/`cmova` chain), a quarter the SSE2 consecutive-element test (`global_run`); `count_access` + `sectors_fast` are 8% of ATAX and 12% of SYR2K, almost all of it the entry of `sectors_fast` and its stride scan (`on_stride`) for warps that are plain strided gathers (ATAX, BICG, MVT, SYR2K lanes are a fixed multiple of 32 bytes apart).
* `simd::warp_rows_kernel()` (`src/simd/simd.cpp` `warp_rows_k`, AVX2; null at `VGPU_SIMD=base|off` or without AVX2, where the old scalar code runs unchanged): one pass, four lanes to a step, builds the 32 addresses and in the same pass finds a lane in the constant window (high half of `address - window` zero), and whether every lane is on the stride of the first two (xor against the expected address). `h_global` takes it for a full warp: a run is "on a stride of `bytes`" (so the separate `global_run` pass is gone for full warps), the batch uses the addresses as built, and a warp that is on a stride is counted from its ends by `count_global_progression` -> `memcount::detail::full_warp_progression_sectors`, which is `lane_progression`'s conditions and `sectors_progression`'s two closed forms (now the shared `progression_sectors`) without the scan; a warp it does not answer goes to `count_access` as before. Partial masks, 16-byte accesses and the PTX engine are untouched.
* Same bits: counters and stdout identical on all 12 programs below (`VGPU_COUNTERS=1`, 1-3 kernels each, 255 and 1000 for needle and hotspot), bench.py counters `identical` in all 26 canary rows.
### Results (server3, one thread, cpu 2, `perf stat` instructions:u and cycles:u, 15 alternating runs, A = b4a51d7f, B = branch; Mann-Whitney p on cycles)
| program | instructions B/A | cycles B/A | p |
|---|---:|---:|---:|
| ATAX | 0.792 | 0.797 | <0.001 |
| BICG | 0.798 | 0.861 | <0.001 |
| MVT | 0.791 | 0.824 | <0.001 |
| GESUMMV | 0.798 | 0.853 | <0.001 |
| SYR2K | 0.763 | 0.796 | <0.001 |
| GEMVER | 0.865 | 0.904 | 0.001 |
| GEMM | 0.848 | 0.870 | 0.004 |
| 3MM | 0.840 | 0.879 | <0.001 |
| 2DCONV (no change expected) | 0.981 | 1.018 (p 0.23); 25 runs 1.010 (p <0.001) | A/A 25 runs 1.008 (p 0.14) |
| hotspot | 0.993 | 1.010 (p 0.25); 25 runs 1.014 (p 0.014) | A/A 1.001 |
| needle | 1.001 | 0.991; 25 runs 1.000 | |
| gaussian | 1.000 | 1.001 (25 runs; a first 15-run set read 1.35 while another job ran) | A/A 1.000 |
A/A (15 runs): GEMM 0.995, SYR2K 0.997, hotspot 1.001; ATAX read 1.083 (p 1.0) in a window with another job on the box. bench.py canaries (13 workloads x SASS and PTX, 15 runs, cpu 2): all within 0.98-1.03; the two below 0.99 are `wmma` PTX 0.983 and `warp` SASS 0.987 (equal instructions, code unchanged: layout), against an A/A of 0.995-1.004 on mem, shared, int_alu and warp.
### Tests (CMake)
* `test_sass_mem_rows` (and `test_sass_mem_rows_simd_base`, the same under `VGPU_SIMD=base`): `warp_rows_kernel_shapes` runs the kernel over every lane the same, strides from 0 to 2^32 either way, carries out of the low half, wraps of the 64-bit sum, rows as a pair / a single row / none, a lane moved into the window alone and a lane off the stride by one bit, against the lane definitions; and the closed form's sectors for each progression against `distinct_sectors_uncapped` of its addresses.
* `test_sass_generic_mem`: `global_accesses_on_a_stride_or_not` runs LDG/STG of 1 to 16 bytes (sign and zero extension, guards) at strides 0 to 200 bytes (runs, broadcast, misaligned, sector-wide) for 32 lanes, 20 lanes and the halves a guard leaves, tier against the interpreter (`VGPU_FASTPATH=0`) and the written-out rule: bytes, registers and every counter.
* Mutations that fail them (each rebuilt and run): the run test `stride == bytes` -> `<=` (generic_mem); the counted lanes 32 -> 31 in `count_global_progression` (generic_mem); the kernel's expected-address step +8 (mem_rows); the window shift 32 -> 33 (both); the hi-row shift 32 -> 31 (mem_rows); the `last > top` wrap guard removed from `full_warp_progression_sectors` (mem_rows).
* Not run: `e2e_sass_tier` / `e2e_sass_archs` / the AMD guards (no nvcc on server3; CI), ASan/UBSan/TSan builds (no float, atomic or threaded code in the change; CI).
### Not done
* The kernel stores the 32 addresses even for a run (where the batch path never reads them); keeping them in registers and storing only for a non-run would remove 8 stores per run warp (hotspot +1.4% and 2DCONV +1.0% cycles at 0.7-1.9% fewer instructions may be that); not tried.

### Part 2: the SASS run loop goes on past S2R, ULDC, UMOV, USEL, PLOP3, SEL and a convergent BRA (branch `sp14-step`)
Profile first (server3, `perf record` cycles and instructions, one thread; A = b4a51d7f). `Runner::step_warp` (the tier's run loop `tier_run_t` is inlined in it) is 17% of the cycles of backprop, 20% of hotspot, 21% of needle, 17% of gaussian, 16% of 2DCONV, 10% of GEMVER, 7-8% of GEMM and ATAX, 5% of lud (lud's top is `h_shared`, 15%). The machine is retire-bound there (hotspot: 67% retiring, 4 ops per cycle, 2.2 instructions per cycle), so instructions retired is the metric that matters. Inside `step_warp` the samples are flat, with two clusters: the per-op wrapper of the loop (steps, counters, guard, the store gate: 25-30 instructions around every handler call; 27-31% of the function's cycles) and the end of a call (the handler of the op that ends it, 32 program counter stores, ~20 loads and branches of the chain, the set-up of the next call: roughly 140-200 instructions per call). `VGPU_TIER_STATS` says which ops end the calls: in short kernels it is the prologue. S2R and ULDC end 74% of gaussian's 60M calls, 71% of 2DCONV's 5.2M, 61% of GEMVER's 4.3M, 42% of backprop's 3.1M; a taken or untaken BRA ends 30% of hotspot's 12.6M. Those ops write registers and nothing else, but are not "straight" (the scheduler must see them, because a diverged warp picks its next group between instructions).
* **The loop goes on** (`MOp::cont`, set in `Code::tier_program` for the forms S2r, Uldc, Umov, Usel, Plop3, Sel; and for a `Form::Bra`): when the chain would have started the next call exactly where the loop is -- the warp has not diverged (`w.runnable() == group`), no yield, slice, give-way, WARPSYNC hold or forced mask, `tier_chain_` on, no statistics, registers inside the file -- and the turn has calls left (`w.budget`), the loop does what the chain does without leaving: `--w.budget; ++w.extra`, the 256-op bound starts afresh (`olim`), the counters of the op are counted as before. A BRA is continued only if the whole group takes it (to an address inside the section) or none of it does; the divergent-branch counter is kept; its spin rule needs lanes outside the group, and there are none. The program counters are not stored for the ops it passes; every way out of the loop sets them where it stops, as it already did after any run that had moved on (the test `idx != first` becomes `o != o0`, and `o0` stays the start of the real call). The call count, the turn limit (4096) and the clock readings (S2R of SR_CLOCK counts `w.steps`) are unchanged.
* **The loop's place is a pointer** (`o`, with `o0` and `olim`) instead of an index and a counter `n` (they moved in step), and the store gate reads `MOp::store` instead of `ops[idx].ins->op` (a dependent load into the large `Instr`, 5-8% of `step_warp`'s cycles in hotspot, gaussian and 2DCONV). Alone this was -1.5% instructions and no measurable cycles; kept because it is small and removes the stack slot the loop carried (`addq $1, -0x170(%rbp)`). A fast path for the common op (a flag `fast`, one test instead of five) was written and dropped: instructions did not go down (0.993 vs 0.986 of A in hotspot).
### Results (server3, cpu 6, one thread, `perf stat` user cycles and instructions, 15 alternating runs, Mann-Whitney p; outputs and counters identical)
| program | cycles B/A | p | instructions B/A | A/A cycles (cpu 10) |
|---|---:|---:|---:|---:|
| rodinia gaussian | 0.917 | 0.010 | 0.880 | |
| polybench 2DCONV | 0.969 | 0.001 | 0.945 | 0.996 |
| polybench GEMVER | 0.969 | <0.001 | 0.966 | |
| rodinia backprop | 0.973 | 0.008 | 0.974 | 1.000 |
| rodinia hotspot | 0.967 | <0.001 | 0.949 | 0.998 |
| rodinia pathfinder | 0.982 | 0.001 | 0.979 | 1.017 (p 0.48) |
| rodinia lud | 0.987 | 0.003 | 0.976 | |
| rodinia needle | 0.990 | 0.005 | 0.988 | 1.003 |
| polybench GEMM (no change expected) | 0.988 | 0.84 | 0.988 | 0.996 |
| polybench ATAX (no change expected) | 0.985 | 0.046 | 0.988 | |
(A/A instructions: 1.0000 in every row. A/A of cycles medians moved up to 1.7% on pathfinder; the box's other jobs add outliers to the spread.)
bench.py canaries (13 workloads x SASS and PTX, 15 alternating runs, cpu 10, counters identical in every row; `winst_per_cpu_s`, median B/A): int_alu 1.048 / 1.012, f32_fma 1.111 / 1.003, f64_fma 1.005 / 1.000, wmma 0.996 / 0.977, mma_ptx 0.999 / 1.018, mem 1.029 / 1.029, shared 1.029 / 0.978, atomics 0.997 / 0.984, local 1.013 / 1.016, divergent 1.008 / 1.007, warp 1.030 / 1.000, launch 1.024 / 1.023, api 0.979 / 0.975 (SASS / PTX). A second run of the rows that read under 0.98, 25 runs, with an A/A beside it (A/A: every row 1.000-1.003): wmma ptx 0.978, shared ptx 0.975, atomics sass 0.981, atomics ptx 0.975, api sass 0.971, api ptx 0.973; wmma sass 0.993 and shared sass 1.029. None of these rows runs a changed instruction (the PTX interpreter, the API path and the tensor handlers are untouched, counters identical): they are code placement, the lottery earlier rounds describe (one LTO unit, so a change in `step_warp` moves what follows it). A control with a trivial unrelated edit (one more `case` in `Runner::sreg`, no behaviour change) against A, same box, noisier hour: mem sass 0.977, mem ptx 0.980, wmma ptx 0.980, all p > 0.2 but the same size. Not fixed: no deterministic placement fix was found in the time of this round, so the canary rule (no row 2% worse) is NOT met on api, atomics and the PTX wmma/shared rows; whether that is acceptable against 3-8% on real programs is for the merge decision.
### Tests (CMake)
* `test_sass_chain_cont` (new): sm_86 SASS written out word by word (the decoder's text is checked against each word), run with the tier and with `VGPU_FASTPATH=0`; words, every counter and the clock readings (CS2R SR_CLOCKLO counts the warp's instructions) must be equal. A diverged warp whose halves meet at an S2R (lead 1/2/5, tail 1/3, one and two warps, part of a warp); two or four warps that update one word 3000 times with 1-5 S2Rs in the loop (where the 4096-call turns end decides which updates are lost); a branch taken by every lane, by none, by half, guard on half with a false condition (`@P0 BRA P1`), 40 threads.
* `test_sass_store_gate` (new, plus a `VGPU_TIER=0` run): four warps load, add one and store one word: the word ends as 1 (the card's answer), whatever the number of warps, with the store 3, 256, 257 or 300 instructions into the kernel.
* `test_tier`: `MOp::store` is exactly "STG or ST" on every instruction of the 11 disassembler corpora, and the corpora hold stores.
* Mutations, each rebuilt and run: no `w.runnable() == group` test -> `halves_of_a_diverged_warp...` fails; no `--w.budget` -> `the_calls_saved...` fails; no `++w.extra` -> `the_calls_saved...` fails; the in-loop divergent-branch count removed -> `a_branch_the_whole_group...` fails; the split-group test of BRA replaced by `true` -> the diverged-halves and the branch tests fail; the BRA target replaced by the next op -> all four tests fail; `MOp::store = false` -> all three store-gate tests and test_tier fail.
* ASan+UBSan (-O0, halt on error) of test_sass_chain_cont, test_sass_store_gate, test_sass_alu_runs and test_tier: pass. Release: test_jit (22), test_jit_resid, test_jit_share, test_sass_block_reuse, _dispatch, _lanes, _generic_mem, _s2r_fill, _ldgsts, _atomics, _mem_rows, _code_at, _access_count: pass; `VGPU_JIT=1 test_sass_chain_cont` passes. Not run here (no nvcc on server3, CI): `e2e_sass_tier` (sass_tier_steps.cu, which tests the same gate and step rules per program), `e2e_sass_archs`, `e2e_sass_jit`, TSan (nothing shared is touched: `Warp::budget` and `extra` are the warp's own).
### Not done, and negative results
* The `fast` flag for the common op (one test in place of straight/fn/unguarded/regs): instructions 0.993 of A in hotspot against 0.986 without it; dropped.
* Not continued past: BSSY and NOP (needle, pathfinder: 0.5M calls of 2.8M), the tier's JIT variant (`Jit` instantiation keeps the old loop shape, only its index became a pointer), SHFL / VOTE / CS2R (not in the list of forms that write registers only, and their calls are rare in these programs).
* Deferring the 32 program counter stores to where the loop stops for the passes that do end (the chain stores them again at once): about 18 of ~150 instructions of a call; not tried.
### Both parts together (server3, cpu 2, one thread, `perf stat` instructions:u and cycles:u, 15 alternating runs; A = main a28b4972, B = this branch; counters and stdout identical on all four)
| program | instructions B/A | cycles B/A | p |
|---|---:|---:|---:|
| polybench ATAX | 0.781 | 0.795 | <0.001 |
| polybench GEMM | 0.837 | 0.889 | <0.001 |
| rodinia hotspot | 0.941 | 0.959 | <0.001 |
| rodinia gaussian | 0.880 | 0.898 | <0.001 |
The gains add: the gather-bound rows keep part 1's gain (ATAX 0.797 alone), the short-kernel rows keep part 2's (gaussian 0.917 alone, hotspot 0.967 alone). Part 2 moves six bench.py canary rows more than 2% the wrong way (`wmma` PTX 0.978, `shared` PTX 0.975, `atomics` SASS 0.981, `atomics` PTX 0.975, `api` SASS 0.971, `api` PTX 0.973; 25 runs beside an A/A of 1.000-1.003) on workloads whose executed code it does not touch; read as code placement in the single LTO unit, not fixed (see the layout notes of rounds sp4 to sp10).

## Round sp16 (AMD): byte and short global accesses of the AMD engine, as runs (branch `sp16-amd`, base 705ed36f)

Profiled first (server3, Zen 4, `perf record -e cycles:u`, one host thread, `bench_amd <workload> full` on the MI300X profile, base 705ed36f). Shares of the cycles, outside the
pre-decoded tier's arithmetic kernels:

| workload | top symbols (share of cycles) |
| --- | --- |
| `narrow` | `wave_scalars` 31.8, `tier_global` 16.5, `MemoryManager::load_batch_in` 7.8, `store_batch_in` 7.5, `batch_addresses_ok` 6.7 |
| `atomics` | `wave_atomics` 30.4, `global_access` 12.9, `rmw_batch_fn` 12.1, `atomic_result` 8.2 |
| `buf` | `buffer_words` 26.0, `batch_addresses_ok` 13.9, `load_batch_in` 13.2, `buffer_access` 10.1 |
| `flat` | `global_access` 18.7, `memory_run` 16.3, `flat_access` 15.6 |
| `mem` | `memory_run` 13.7, `h_l64_avx2` 13.2, `run_groups` 10.6, `wave_scalars` 8.9, `tier_global` 8.0 |
| `lds`, `ldsx`, `wave` | `h_ds_avx2` 20 to 27, `run_groups` 17 to 25, `memory_run` 12 to 18 |
| `divergent`, `smem` | `run_groups` 25 and 27, `h_mad_u64_avx2` 12 (divergent) |
| `valu_*` | the host program's own check (`main`, libm) is 43 to 69%; the engine is 22 to 31% |

`wave_scalars` is the path for a byte or a short a lane: the addresses of the active lanes are collected, `load_batch` / `store_batch` make the accesses together, and the values go back
lane by lane. A word or more a lane with consecutive addresses already went to memory as one call (`memory_run`: `load_run` / `store_run`); a byte or a short never did, though a
`char` or `half` array read a lane at a time (images, strings, quantized weights, fp16 loads) is exactly consecutive addresses.

### What changed

`Machine::narrow_run` (`amd/src/gcn_exec.cpp`, out of line as `memory_run` is): the lanes from the current one that are active and take addresses a byte (or a short) apart go to memory as one
`load_run` / `store_run` with the element as the unit, and a load widens each element into its lane (zero- or sign-extended). It declines, having done nothing, where `memory_run` does (fewer than two
lanes, a fault armed, a host mapping, a run not inside one allocation, a misaligned run), and the lanes then go to `wave_scalars` and one at a time as before. The interpreter's `global_access` calls it
for the plain-register forms; the tier's byte and short accesses moved out of `tier_global` into `tier_narrow` (`amd/src/gcn_tier_machine.inc`), so the word loop of `tier_global` has only lost its sub-dword
branches. The load memo's rule (a lane that reads the address the lane before it read takes that value) is kept: a run does not start on such a lane, and the last element of a run is the next lane's memo.
Same bits: results, error text and counters do not change.

### Results (server3, `perf stat` user cycles and instructions, cpu 3, 15 alternating runs, A = 705ed36f built from this tree before the change, B = branch; `VGPU_THREADS=1`, `full` size)

| workload / target | instructions B/A | cycles B/A | p |
| --- | ---: | ---: | ---: |
| `narrow` MI300X | 0.498 | **0.505** | <0.001 |
| `narrow` RX 6800 | 0.713 | **0.774** | <0.001 |
| `mem`, `flat`, `buf`, `atomics`, `lds`, `ldsx`, `divergent`, `wave`, `valu_int`, `valu_f32`, `smem` (MI300X) | 0.998 to 1.001 | 0.970 to 1.012 | `atomics` 0.970 and `ldsx` 1.012 (p < 0.01), the rest p > 0.01 |
| the same eleven on the RX 6800 | 1.000 to 1.007 | 0.934 to 1.031 | see below |

Rows that run no narrow instruction and still moved on the RX 6800: `mem` 1.031 and `lds` 1.026 (equal instructions to within 0.5% and 0.7%), `flat` 0.957 and `atomics` 0.934. Four code
placement results of the library in both directions, as the sp13 and sp14 rounds saw for other changes (the A/A of the same binary against itself reads 0.992 to 1.004 on the rows tried), so they are not a claim either way.
`bench.py --ab`'s counters-identical check says `yes` on the 24 rows of the first A/B (`/tmp` run of 12 workloads, both targets).

### Tests (CMake-wired)

* `test_amd_gcn_narrowruns` (new; kernels `amd/tests/data/narrowruns.c`, built for gfx942, 1030, 1100, 1201 and 1250 and run on all five, and once more with `VGPU_AMD_TIER=0` for the interpreter's own path): ten copy
  and widening-load kernels (byte and short, signed and not, 64-bit and 32-bit offsets) over 18 address patterns (consecutive, reversed, strided, repeated, overlapping stores, misaligned, over chunk and allocation edges, off the end,
  wild, random) and 7 lane masks, 3 draws each; the run path on and off against twin memories: memory images, resident bytes, error text, wave / instruction / VMEM / flat counts. And a pair of directed cases: sixteen lanes
  storing a byte to one address (the last lane stays), and a signed load of bytes with the top bit set. The five code objects are in `codeobject_digests.txt` (`test_amd_codeobject_same_bits`).
* Mutations of `narrow_run` (rebuilt and run on server3; each fails the test): sign extension ignored (both targets, 0 of 2 pass); a short stored as one byte (1 of 2 pass); the run not stopping at an inactive lane
  (fails on gfx1030, passes on gfx942: the wave32 masks catch it); the next lane's memo taking the first element (1 of 2).
* Also run and passing: `test_amd_gcn_memruns`, `test_amd_gcn_memory_same_bits` (tier on and off), `test_amd_gcn_same_bits`, `test_amd_gcn_tier_same_bits`, `test_amd_gcn_narrow`, `test_amd_gcn_memory`,
  `test_amd_codeobject_same_bits`.

### Not done

* `wave_atomics` + `rmw_batch_fn` + `atomic_result` are 51% of `atomics`: the per-lane callback does a switch on the operation each lane; a batch specialised by operation (add, min, max, and, or, xor) would drop it. Not tried.
* `buffer_words` (26% of `buf`) reads each lane's offset and bounds through generic code; raw buffer accesses with a plain-register offset have no run path. Not tried.
* `run_groups` (the wave stepping loop, 10 to 27% of the memory and control rows) is where the per-instruction overhead of the interpreter sits, as `step_warp` is for SASS; the sp14 round's work on the SASS run loop has no AMD twin yet.
* ASan, UBSan and TSan were not run on this change; it touches no threaded or atomic code, and the buffers are fixed-size stack arrays of twice the wave size.

## Round sp22 (AMD): radixsort2, uniform loads and `v_bfrev_b32` (branch `sp22-amd`, base `origin/main` 85ec63a5)

Program: HeCBench `radixsort2` hipified and built for gfx942 (`hipcc -O2`; `-n=65536 -float -keysonly`, one `thrust::sort` through rocPRIM, 700 dispatches, 322 M wave instructions, 5.3 s on `amd/mi300x`, one thread). Profiled with the `LD_PRELOAD` `SIGPROF` sampler (`tools/bench-amd/profile`), `-g` build, Zen 5 (server2, cpu 13, load 1 to 7).

### What the profile said, and what changed
1. `v_bfrev_b32` (1.3 M executions, 13 % of the program) was not a tier handler: the interpreter reversed the bits with a 32-iteration loop per lane. Now a vector kernel (`k_bfrev`, swap neighbours, pairs, nibbles, then `bswap`) for the `_e32`, `_e64`, SDWA and DPP forms (`amd/src/gcn_tier.cpp`, `gcn_tier_kernels.inc`). 5.31 s -> 4.26 s.
2. `v_cndmask_b32` with every lane active selects straight into the destination row (it went through a 64-word temporary and a row copy); a partial EXEC keeps the old path. 4.26 s -> 4.11 s (the handler's own samples fell 488 -> 353; a first version that also did the partial case in the same pass made the program 4 % slower at the same counters, its cost moved to `h_l64` and `l_lshl_add_g`: the placement lottery, so only the whole-wave case is kept).
3. `Machine::wave_scalars` (`amd/src/gcn_exec.cpp`): a load whose lanes are all on one address (a uniform value read through vector registers, most of this program's loads) reads that address once and broadcasts it (`exec::amdrun::all_equal`, SSE2, in `include/vgpu/exec/amd_run_scan.hpp`); the repeated-address copy loop (a lane that follows a lane with the same address takes that lane's value) ran for every load, was 10 % of the program as a branchy dependent chain, and is now only run when `conc()` (on one host thread nothing writes between two lanes' reads of an address, so the values already agree), and then only if `any_adjacent_equal` finds a pair. 4.11 s -> 3.5 s.

Result (user time, 9 alternating runs, cpu 13; A = `origin/main`, A2 = a copy of A's build): A 5.40 s (5.31 to 5.54), A2 5.48 s, W 3.55 s (3.47 to 3.61): **0.657**. Counters (`VGPU_DISPATCH_STATS`, every dispatch line) and the program's output are identical on A and W for three argument sets (`-n=65536 -float -keysonly`, `-n=20000`, `-n=30000 -float`); `bench-amd --check-counters`: 66 workload/target pairs match; radixsort2 passes with `VGPU_THREADS=4`. Not done: `ds_read2_b64` / `ds_write2_b64` (0.4 M executions, `lds_access` 4 %) are still interpreted, and what is left is spread (`run_groups` 23 %, `h_cndmask` 10 %, `load_batch` 7 %).

### The two sp21 canary rows
`p_smem_wide` (0.943) and `narrow` (0.970) were reported lost at equal counters when sp21's `v_div_scale` change landed. They are not reproducible here: a fresh build of `origin/main` + that commit (`Ad`) against `origin/main` (`A`), 15 runs, gives `p_smem_wide` 1.003, `narrow` 1.008. Nothing to recover, and no layout change was made (a fix cannot be measured where the loss is not). With this branch on top of sp21's commit (`Wd`) against `Ad`: `p_smem_wide` 0.989, `narrow` 0.998, but `smem` 0.953 and `wave` 0.921 (A against `Wd`: 0.945 and 0.938; the A/A of a copy: `smem` 0.999, `wave` 0.977), at equal counters. Without sp21's commit (`W` against `A`) `smem` is 0.981 and `wave` 0.997. So when #469 and this land together `smem` and `wave` may lose 5 %: it is the placement of `gcn_exec.cpp` (the edit in `wave_scalars` moves everything behind it); re-measure after the merge.

### Canary, this branch alone (mi300x, 15 runs, one thread, cpu 13; W/A, with the A/A of a copy in brackets; counters equal)
`atomics` 1.035 (1.030), `divergent` 0.982 (1.005), `flat` 0.979 (1.000), `mem` 1.011 (0.996), `mfma` 0.992 (1.002), `mfma_fp8` 1.027 (1.001), `narrow` 0.988 (0.996), `smem` 1.043 (1.059), `smfmac` 0.990 (1.002), `valu_int` 1.000 (0.988), `wave` 0.977 (0.990), `p_cmp` 1.028 (0.988), `p_dpp` 0.970 (1.003), `p_dpp_arch` 0.980 (0.996), `p_lane` 1.067 (1.048), `p_perm` 0.928 (0.951), `p_sbuf` 0.988 (1.058), `p_smem_wide` 0.986 (0.967); the launch rows and the rest within 0.97 to 1.04. Outside the A/A: `p_dpp` -3 %, `divergent` -1.8 %, `flat` -2.1 %. `lint_hot_unit_layout` passes.

### Tests and mutants
* `test_amd_gcn_tier_same_bits` (tier against interpreter): `amd/tools/tier-corpus.py` now writes `v_bfrev_b32` in every form (e32, e64, SDWA, DPP, DPP8) and the six `isa_corpus_tier_*.txt` files were regenerated with llvm-mc (54 lines added to the CDNA ones, 32 to the RDNA ones, nothing else changed). Mutants: no `bswap`: killed; the pair-swap step dropped: killed; `v_cndmask` taking the whole-wave path for a partial EXEC: killed; `v_cndmask` with the sources swapped: killed.
* `test_amd_run_scan`: `all_equal` and `any_adjacent_equal` against their loops (every length, one entry different in either word at every position, random lists). Mutants: `all_equal` without its scalar tail: killed; comparing the low words only: killed; `any_adjacent_equal` without its tail: killed; either half instead of both: killed.
* `test_amd_gcn_gather_same_bits`: new flavour 25 (every lane at one address, any word; the corpora's every flat/global instruction). Mutant: the broadcast writing 0: killed only with flavour 25 (it survived the old flavours: they never make a whole wave one address); the all-equal test always true: killed. Survived, equivalent: the `conc()` guard of the repeated-address copy (the tests run one thread, as the sp18 notes say for the same rule).

## Round sp18 (SASS): the global-memory path (branch `sp18-sass`, base `sp17-speed` dd3adb43)

The question: what does the SASS tier spend its instructions on in the gather-bound real programs, and in the issue loop? Measured first, on server3 (Zen 4), with `perf record -e instructions:u -c 1000000` on one thread (`VGPU_THREADS=1`, SASS, `nvidia/rtx3060`), PolyBench/GPU and Rodinia from the Round sp8 corpus, and the canary workloads with `-c 2000000`.

### What the profile said (base build, share of instructions retired)
* ATAX: `warp_rows_k` 24%, `load_batch` 14%, `h_global<false>` 9%, `load_broadcast` 8%, `store_run` 6%, `load_run` 6%, **`find_ordinary_miss` 5%**, `count_global_progression` 5%, `step_warp` 3%. SYR2K: `load_batch` 26%, `warp_rows_k` 17%, `h_global<false>` 8%, `count_global_progression` 8%, `load_broadcast` 7%, `find_ordinary_miss` 2.5%.
* `warp_rows_k` (the full warp's addresses, window and stride tests) was 166 instructions with six 32-byte spills to the stack (the chain of eight expected-address vectors the compiler hoisted).
* `count_global_progression` was a function of its own: six pushes and pops around a dozen operations, and `memcount::fast()` read through the GOT (two loads of a guard and a flag) on every counted access.
* `find_ordinary_miss`: a kernel that reads A, then x, then writes y asks the thread's lookup memory for three allocations in turn, and the lookup tries the way that answered *last* first, which is the allocation of the instruction before. So most accesses of such a kernel took the miss path (a call, a loop over the ways, the epoch tests). Each global instruction reads the same allocation every time.
* The issue loop (`step_warp`, which holds `tier_run_t`): 9% of `int_alu`, 27% of `warp`, 36% of `divergent`, 45% of `shared` (barrier turns), 3 to 6% of the real programs, and **flat**: in a 100,000-instruction sample of `shared` no instruction of `step_warp` has more than 5.6% of the function's samples (the top one is the epilogue's `pop`s: a call per warp turn), the rest is a few percent each. The per-instruction `++st.instructions; st.thread_instructions += lanes; st.inst_by_class[cls] += lanes` and the `w.steps` update are memory read-modify-writes, but `w.steps` is read by S2R CLOCK in `exec.cpp` and the counters by the fallback handlers, so keeping them in registers needs a flush before every call out of the loop. Not attempted (see Not done).

### What changed
1. **A hint for the allocation lookup** (`MemoryManager::find_ordinary(addr, len, base, hint)`; `MOp::way`). Each global op keeps the way of the lookup memory its allocation was last found in and passes it (`&op.way`) to `load_batch`, `store_batch`, `load_broadcast`, `store_strided`, `load_run`, `store_run`; the lookup asks that way first instead of `mem.last`. The hint is only where to look first: any byte (masked to the ways), none, another thread's (read and written with relaxed atomics), stale after a free (the epoch test still decides). The thread's `last` and `prev` ways are not touched by a hit.
2. **`warp_rows_k` without the spills**: each group of four addresses is compared with the one before it plus `4 * stride` (the first with `a0 + 0..3 * stride`, by starting `prev` one step back) instead of with a chain of expected addresses; the constant-window test takes the unsigned minimum (`vpminud`) of `(a - window)` over the groups and looks at the high halves once at the end instead of compare-and-or per group; the three shapes (`lo` and `hi` rows, `lo` only, neither) are three instantiations, not tests in the loop. About 165 to about 115 instructions, no spill.
3. **`count_global_progression` inlined** into the handler, with `fast` the launch's own copy of `VGPU_FASTCOUNT` (`Runner::fastcount_`, sampled at launch).
4. `lookups()` (the thread's lookup memory, found by the thread pointer in a table of slots): the first probe inline, the rest in a function of its own (it was a call and a loop set-up on every lookup). `load_broadcast` fills its 32 entries with a counted loop of fixed length (sixteen stores).

### Results (counters and outputs identical in every run)
Zen 4 (server3), one thread, `perf stat -e instructions:u,cycles:u`, alternating A B, medians, Mann-Whitney p on cycles; A = base build (dd3adb43), B = this branch, each built here and frozen. CPU 13 for the first table (its SMT sibling was used by another session during parts of the run: the A/A of that window drifted to 1.04 on ATAX, so the rows were repeated on CPU 4 with an idle sibling: ATAX 0.866, GEMM 0.916, BICG 0.926, MVT 0.901, the same as below).
| program | runs | instructions B/A | cycles B/A | p |
|---|---:|---:|---:|---:|
| polybench ATAX | 9 | 0.819 | 0.864 | <0.001 |
| polybench BICG | 9 | 0.850 | 0.930 | 0.019 |
| polybench GESUMMV | 9 | 0.877 | 0.924 | 0.031 |
| polybench MVT | 9 | 0.834 | 0.894 | 0.005 |
| polybench GEMM | 9 | 0.833 | 0.913 | <0.001 |
| polybench SYR2K | 5 | 0.887 | 0.914 | 0.009 |
| polybench SYRK | 5 | 0.920 | 0.924 | 0.009 |
| polybench 3MM | 5 | 0.844 | 0.919 | 0.009 |
| polybench CORR | 5 | 0.956 | 0.937 | 0.60 |
| polybench 2DCONV (no change expected) | 9 | 0.987 | 0.991 | 0.07 |
| polybench JACOBI2D | 9 | 0.988 | 0.975 | <0.001 |
| rodinia needle | 9 | 1.000 | 0.981 | 0.004 |
| rodinia hotspot | 9 | 0.997 | 1.007 | 0.002 |
| rodinia gaussian | 9 | 0.994 | 0.982 | <0.001 |
| rodinia pathfinder | 9 | 0.999 | 0.988 | 0.97 |
| rodinia backprop | 9 | 0.977 | 1.000 | 0.97 |
| rodinia lud | 9 | 0.998 | 0.996 | 0.024 |
| polybench GRAMSCHM (702 G cycles a run) | 3 | 0.861 | 0.969 | 0.05 |
| polybench FDTD-2D (494 G cycles a run) | 3 | 0.972 | 0.977 | 0.13 |
A/A (A against itself): ATAX 0.999, GEMM 0.999 (CPU 4, 9 runs); on CPU 13 (sibling busy) needle 1.000, hotspot 1.001, 2DCONV 1.004, GEMM 1.001, ATAX 1.041.

`bench.py --ab` canary (13 workloads x SASS and PTX, 9 runs, CPU 4, load 8 to 11 from other sessions): counters `identical` in all 28 rows; the only row the script calls different is `mem` SASS 1.052x (p 0.024; its A/A is 1.007x); every other row is inside its own A/A spread (0.97 to 1.03): `divergent` SASS 1.019x (A/A 0.989x), `shared` SASS 1.016x, `local` 1.010x, `int_alu` 1.010x, `api` SASS 1.017x (A/A 0.987x), `launch` SASS 1.002x. The PTX rows (1.00 +- 0.02) run the same memory manager with no hint.

Step by step (single runs, instructions B/A on ATAX / GEMM / SYR2K): the new `warp_rows_k` alone 0.978 / 0.978; plus the inlined counting, the inline first probe and the broadcast fill 0.939 / 0.948 / 0.942; plus the hint 0.819 / 0.833 / 0.887.

### Tried and not kept
* `-fno-semantic-interposition` on the hot units (the memory manager's functions are called through the PLT inside the library, and GCC will not inline an exported function): instructions -0.4% on top of the above, cycles +4.4% (ATAX), -1.4% (GEMM), +4.5% (SYR2K): a layout re-roll, not a gain. Dropped.

### Tests (CMake-wired, no GPU) and mutations
* `test_memfast`: `a_lookup_hint_is_only_a_place_to_look_first` (random, stale, out-of-range and no hints for `load_batch`, `load_broadcast` and `load_run` over twelve allocations (more than the eight ways) against the words stored; every 700 iterations an allocation is freed and the calls with every hint must decline its addresses, then a new one is made) and `more_threads_than_lookup_slots_all_read_the_right_words` (1,300 threads alive at once, over the 1,024 slots, each reading through the scalar, batch and broadcast paths).
* `test_sass_mem_rows` (existing, holds the AVX2 kernel to the lane definitions over 3 kinds of rows x 34 positions of a lane in the window or off the stride x 8 adds x 12 bases x 16 strides) also with `VGPU_SIMD=base`; `test_warp_stride_access` (broadcast counts 1 to 32).
* Mutants built and run on server3 (each killed unless noted): window minimum taken as a maximum; `prev` not advanced; `prev` not started a step back; step doubled; window test without the shift (and with the wrong shift); high half shifted by 31; broadcast fill one lane short; first probe inverted; hint hit skipping the epoch test; hint hit returning a wrong base. Survived, equivalent: the group difference written as a subtraction instead of an xor (same zero test); the probe loop returning the slot of the first probe instead of the probed one (another thread's lookup memory: the answers are the same and only a data race, which TSan, not run here, would flag); the hint not masked to the ways (reads past the entries: caught only by ASan, not run here).
* Also run, all pass: `test_tier`, `test_sass_alu_runs`, `test_sass_atomics`, `test_sass_block_reuse`, `test_sass_dispatch`, `test_sass_generic_mem` (and with `VGPU_SIMD=base`), `test_gather_sp10`, `test_memhot_sp8`, `test_memcopy_direct`, `test_sass_access_count`, `test_memrmw`, `test_memalloc`, `test_sass_ldgsts`, `test_hostmap_cache`, `test_exec`, `test_exec2`, `test_fastpath`, `test_sched`, `test_sass_lockfree`, `lint_hot_unit_layout`. Sources compile with g++-12 (`-fsyntax-only`). Not run: the sanitizer builds and the AMD guards (CI).

### Not done
* The issue loop: its profile is flat and the per-instruction counters need a flush protocol around every call out of the loop (S2R CLOCK reads `w.steps`; the fallback handlers read the counters). Hoisting `tier_stats_` and `max_steps_` into locals was not tried: the loop has no spare register (its comment says so).
* Calls inside the library go through the PLT (`batch_faults_clear`, `find_ordinary_miss`, `host_map_free_span`'s callees): `-Wl,-Bsymbolic-functions` would remove it but changes which copy of a function the shim libraries that each carry the core bind to; not looked at beyond the flag above.
* `load_batch` (13 to 26% of the gather rows) is 14 instructions a lane and bound by the L1 misses; `load_broadcast` is still about 110 instructions a call (the fault checks, the lookup, the chunk test, the fill).
* `POPC` calls libgcc's `__popcountdi2` per lane (3% of `int_alu`): the AVX2 unit is compiled with `target("avx2")` and no `popcnt`.

## Round sp21 (AMD): float division on real HIP programs (branch `sp21-amd`, base `origin/main` 179c886f)

Question: where does the AMD tier spend its time on real HIP programs, beyond PolyBench? Built for gfx942 with `hipcc -O2` after `hipify-perl`: 18 of 22 HeCBench programs (the reduced set's `-cuda` sources; `attention` needs `cooperative_groups/reduce.h`, `bh` a 64-bit `__shfl_sync` mask, `black-scholes` and `s3d` do not hipify cleanly) and 8 Rodinia apps (hotspot, gaussian, needle, pathfinder, bfs, lud, nn, srad; backprop's C files do not compile as hip). Profiled with the `LD_PRELOAD` `SIGPROF` sampler (`tools/bench-amd/profile`, `SIGPROF_STACK=1` for callers), one thread, `amd/mi300x`. Most of them finish in under a second at their reduced sizes; the ones with seconds in them:

| program | top of the profile (self time) |
|---|---|
| bilateral | `h_div` 28 %, libm `ldexpf` 19 %, `modf` 14 %, `ilogb` 4 % (the libm entries are called from `h_div`: 265 of 342 `ldexpf` samples, all of the `ilogb`) |
| aobench | `h_div` 19 %, `h_f32_avx2` 17 %, `ldexpf` 10 %, `run_groups` 12 % |
| aidw | `h_div` 14 %, `h_f32_avx2` 13 %, `h_pk` 12 %, `ldexpf` 10 %, `modf` 6 % |
| radixsort2 | `vector_alu` lambda 17 %, `run_groups` 15 %, `wave_scalars` 14 %, `h_cndmask_avx2` 8 %, `load_batch` 7 % |
| shmembench | `h_ds_avx2` 38 %, `ds_run_read` 20 %, `run_groups` 19 %, `ds_run_write` 19 % (already SIMD; a bandwidth test) |

### What changed
`v_div_scale_f32/f64` (`div_scale` in `amd/src/gcn_divide.hpp`, run per lane by the tier's `h_div`) classified its operands with `fpclassify`, `isfinite`, `ilogb` and `scalbn` and scaled with `ldexp`, all libm calls: every float division is two `v_div_scale`. When S1 and S2 are normal numbers (the usual case) every test is now a test of the exponent and mantissa fields (`div_scale`), and the scaling by 2^+-64 / 2^+-128 moves the exponent field of a normal number (`scale_by_pow2`; zero, denormals, infinities, NaNs, or a result out of the normal range go to `ldexp`). Any other operand takes the old code, kept as `div_scale_reference`. The interpreter (`gcn_exec.cpp`) calls `div_scale_reference`: it stays the independent oracle for the tier's same-bits tests, and its machine code is what it was (editing the inlined code of `gcn_exec.cpp` re-rolled the layout of the whole AMD canary: first attempt with the interpreter on the new code lost 3 to 6 % on `valu_int`, `smem`, `mfma_i8`, `p_smem_wide`, `launch_*`).

### Results (Zen 5, server2 cpu 13, `VGPU_THREADS=1`, user-time median of 9 alternating runs, A = origin/main, W = this branch, counters equal, program output (stdout without timing lines, and files) equal in every run)
| program | A (s) | W (s) | W/A |
|---|---:|---:|---:|
| silu | 0.78 | 0.38 | **0.487** |
| bilateral | 2.00 | 0.96 | **0.480** |
| aobench | 0.90 | 0.61 | **0.678** |
| aidw (10 0 1) | 41.7 | 32.0 | **0.767** |
| hotspot (512) | 0.15 | 0.11 | 0.733 |
| lud (1024) | 0.76 | 0.74 | 0.974 |
| gaussian, needle, srad | 0.43, 0.24, 0.08 | same | 1.000 |
| radixsort2 | 5.36 | 5.48 | 1.022 |
| shmembench | 3.12 | 3.11 | 0.997 |
Rows under 0.3 s time start-up as much as the tier; hotspot's 0.15 s is within a couple of timer ticks of 0.11 s, not a claim. radixsort2 and shmembench use no division and are the controls (no A/A copy was timed for the programs).

`bench-amd` canary (mi300x, 44 rows, one thread, 15 runs, cpu 13; `--check-counters`: 66 workload/target pairs match): no division in the canary, so it only measures layout. W/A: `mfma` 1.012, `mfma_i8` 0.978 (A against a copy of A: 0.987), `smfmac` 1.006, `narrow` 0.970 (A/A 1.007), `p_smem_wide` 0.943 (A/A 0.989; stable in a re-run, 0.944), `mem` 0.992, `smem` 1.008, `flat` 1.003, `atomics` 0.984 (re-run 1.008), `divergent` 0.996, `valu_int` 0.992, `p_sbuf` 1.040 (A/A 0.960..1.012), `p_lane` 0.996; the other rows within 0.99..1.03. Two rows lose at equal counters and are not recovered: `p_smem_wide` -5.6 % and `narrow` -2 %, which moved with a no-op change to which unit holds the division code (a `noinline` wrapper of `div_scale` in the tier changed neither) and are the placement lottery of Round sp14 again. `lint_hot_unit_layout` passes.

### Tried and dropped
`v_ldexp_f32` (`k_fldexp`, `__builtin_ldexpf` per lane; 3 % of bilateral): the same bit trick (no libm call for a normal number that stays normal) made it 2 to 3 % faster there and `p_lane` 4.5 % slower at equal counters, from the placement of the handlers behind it in `gcn_tier.cpp`; not kept. Tier-mutants of that version (bound `<= 255`, shift 22) were killed by `test_amd_gcn_tier_same_bits`.

### Tests and mutants
`test_amd_gcn_divide` (existing, in CI): `a_float_div_scale_is_the_reference_for_every_operand`, `a_double_...` (the edge values squared, random operands of every exponent, exponent differences at the quotient's two ends, denormals; S0 each of the denominator, the numerator and a third value; the value's bits and the condition equal to `div_scale_reference`) and `scaling_by_a_power_of_two_is_ldexp` (every edge value and random ones, k from -300 to 300). The reference is the old code, so no golden value is recorded. The tier's `test_amd_gcn_tier_same_bits` (interpreter on the reference, tier on the new code, every `v_div_scale` of the corpora) passes. Mutants of the fast path, each built and run against the test: `d1 > -kMinExp` to `>=`: killed (2 tests fail); `e2 - e1 >= kFar` to `>`: killed; `diff < kMinExp` to `<=`: killed; mantissa test `<` to `<=`: killed; `e2 <= kTiny` to `<`: killed; the exponent range test of `scale_by_pow2` `< kMaxExp` to `<=`: killed (3 fail); `>= 1` to `>= 0`: killed (3 fail).

### Not done
* `radixsort2`'s `vector_alu` lambda / `wave_scalars` / `run_groups` share (the next biggest real-program cost seen) and the other division instructions (`v_div_fmas`/`v_div_fixup` are cheap, `scaled_fma` only runs with VCC set).
* RX 6800 (gfx1030) builds of the programs were not timed; `div_scale` is shared, so the effect is the same code.

## Round sp19 (AMD): the LDS-load double free, and gathered dwordx3/dwordx4 stores on AMD (branch `sp19-amd`, base `sp18-amd` bf4e7622)

### A correctness bug first
`VGPU_GATHER_ONLY=dwordx4 test_amd_gcn_gather_same_bits` aborted with "double free or corruption". Root cause: `global_load_lds` (gfx950's `global_load_lds_dwordx4` and its siblings) checked `at + bytes > lds.size()` with `at = M0 + offset + lane * bytes` in 64 bits. The corpus line `global_load_lds_dwordx4 v[2:3], off offset:-16` with M0 = 0 gives `at = 2^64 - 16` on lane 0, the sum wraps to 0, the check passes, and the `memcpy` wrote 16 bytes in front of the LDS vector's storage: heap metadata, found by the next `free`. The filter only exposed it because the instructions before it in the corpus left the heap in a state where the damage was caught. It is now `at > size || bytes > size - at` (same in `scratch_load_lds`, where the same offset also feeds `scratch_at` first, so the wrap is not reachable there by a negative offset; changed for the same reason, not separately tested). Test: `a_load_into_lds_below_address_zero_is_refused` in `test_amd_gcn_gather_same_bits` (flavour 16, every register zero) reads the refusal from the error text, because the corruption itself is silent. Mutant (check put back to `at + bytes > size`): killed (the instruction is then not refused). The whole unfiltered test and `VGPU_GATHER_ONLY=dwordx4` pass.

### Gathered dwordx3 / dwordx4 stores
`Machine::wave_store_words` (defined after the class, as `wave_load4` is): a store of three or four words a lane whose lanes make no run went lane by lane through `store_words`. Now one `MemoryManager::store_batch` of 4-byte stores in lane order (lane 0's words, lane 1's, ...), single host thread only (`conc()` false). Lane order is kept, so lanes whose quads overlap leave the same bytes whatever the access sizes were; a declined batch (fault armed, host mapping, address off a word or outside the first allocation) has changed nothing and the lanes go one at a time as before.

Results (counters identical, Zen 5, server2 cpu 13, one thread, A = `sp18-amd` + the fix, W = this branch; load 4 to 5). Microbenchmark `x4s` (16K threads x 2048 scattered `float4` stores, 16-byte slots for variant 0, 4-byte-aligned addresses for variant 1; the final memory hash is equal on A and W): gfx942 variant 0 0.52 s -> 0.27 s (0.52x), variant 1 0.56-0.64 s -> 0.31 s (0.52-0.55x); gfx1030 (`amd/rx6800`) 0.57 -> 0.32 s and 0.60 -> 0.36 s. No real program of the PolyBench set stores quads that make no run, so none moves; the claim is for kernels that scatter `float4` or `float3` results (the loads of the same shapes were Round sp18).

`bench-amd` canary (mi300x, 44 rows, one thread, 15 runs, cpu 13): counters identical in all rows (`--check-counters`: 66 pairs match). W/A, with an A/A of a copy in brackets: `mfma` 1.031 (1.003), `smfmac` 1.042 (1.031), `mfma_fp8` 1.063 (1.001), `mfma_bf16` 1.030 (1.002), `flat` 1.028 (0.984), `valu_int` 1.023 (1.011), `mem` 1.002 (0.984), `divergent` 1.003 (0.943), `atomics` 0.976 (1.008, p 0.005), `smem` 0.983 (0.997), `p_smem_wide` 0.966 (0.963), `p_sbuf` 0.995 (1.000); the launch rows swing 0.94 to 1.08 between two copies of A and are not claimed. The only W-specific loss is `atomics` at -2.4 %. This is the layout lottery again (`gcn_exec.cpp` is edited): the rows that Round sp17 lost on Zen 5 (`mfma`, `smfmac`, `divergent`) are at or above 1.00 against `sp18-amd`; `lint_hot_unit_layout` passes.

### Tests and mutants
`test_amd_gcn_gather_same_bits` (flavours 21 to 24 run every `global_store_dwordx3/x4` and `flat_store_dwordx3/x4` of the five corpora, lane by lane against the tier with batches on) passes, as do `test_amd_gcn_tier_same_bits`, `test_amd_gcn_same_bits`, `test_amd_gcn_memory_same_bits`. Mutants of `wave_store_words`, each run against the test: lanes in reverse order: killed (2288 runs differ); the last word of each lane dropped: killed (2978); starting from lane 0 instead of `from`: survived, equivalent (lanes before `from` are stored again with the same data before any later lane, so the final bytes are the same); not testing `conc()`: survived (the test runs one thread).

### Not done
Item 3 (the AMD rows lost to layout on Zen 5) was not attempted: with A/A swings of 2 to 6 % on `divergent`, `launch_*` and `p_lane` between copies of the same library, nothing cheap can be told from noise on this box.

## Round sp18 (AMD): NaN lanes and gathered quad loads (branch `sp18-amd`, base `sp17-speed` dd3adb43)

Two of the three items Round sp17 left (`run_groups` was looked at again and left alone: about 45 instructions of bookkeeping around each handler call, spread over four read-modify-write counters, the state checks of `tier_step` and the `Inst::next` chain, nothing small and bit-safe; the `divergent` row says what an edit near it costs in layout). Measured with a sampling profiler (an `LD_PRELOAD` `SIGPROF` sampler resolved against `nm`; `perf` is not allowed on server2) on the 19 PolyBench/GPU programs hipified and built for gfx942 (`hipcc -O2`, dimensions capped at 512 / 160 / 1536, one thread, `amd/mi300x`).

### What changed
1. `fix_nan32` (`amd/src/gcn_tier.cpp`; 13.5 % of GRAMSCHM, 9.4 % of LU): a lane whose result is a NaN was recomputed by the interpreter's rule, a switch on the rule per lane and a `std::fma` call even where an operand is a NaN, which already says the answer (the first NaN operand in the rule's order, quieted). Now one template per rule, a row pass (32 or 64 lanes, a constant, so GCC makes vectors of it) that does the lanes with a NaN operand without computing anything, and the old per-lane rule only for a lane whose NaN the operation itself made (`inf - inf`, `0 * inf`), and only if there is one.
2. `Machine::wave_load4` (`amd/src/gcn_exec.cpp`, defined after the class so no member of the machine moves; hooked in `tier_global`): `global_load_dwordx4` / `flat_load_dwordx4` whose lanes make no run (a gather of float4s, or quads on a 4-byte boundary) went lane by lane through `load_words` and `load_quad`. Now two batches, as `load_wave` and `wave_load3` do: a lane on an 8-byte boundary is two 8-byte loads (what a quad on a 16-byte boundary and the two pairs of a 8-byte one give), any other lane an 8-byte load in the middle and a word at each end. The quad's 16-byte access becomes two 8-byte ones, which only a thread writing the quad at that moment could tell, so the batch is made only when `conc()` is false. Stores of the same shapes are unchanged (they must keep lane order across mixed sizes).

### Results (counters identical; Zen 5, server2, cpu 13, `VGPU_THREADS=1`, user-time median of 9 runs, A = `sp17-speed`, W = this branch, A2 = a copy of A in the same window; load 4 to 5)
| program | W/A | A2/A | | program | W/A | A2/A |
|---|---:|---:|---|---|---:|---:|
| GRAMSCHM | **0.858** | 0.997 | | GEMM | 0.997 | 0.994 |
| LU | **0.913** | 0.990 | | 3MM | 0.975 | 1.000 |
| X4 (float4 gathers, below) | **0.410** | 1.023 | | 2MM | 0.981 | 0.995 |
| SYR2K | 0.995 | 1.003 | | SYRK | 0.997 | 1.000 |
| CORR | 0.994 | 0.994 | | COVAR | 0.989 | 1.004 |
| FDTD-2D | 0.994 | 0.986 | | JACOBI2D | 1.001 | 1.001 |
| 2DCONV | 1.027 | 0.994 | | 3DCONV | 0.989 | 0.994 |
| ATAX | 0.982 | 0.984 | | BICG | 0.955 | 0.988 |
| GEMVER | 0.978 | 0.998 | | GESUMMV | 1.025 | 0.961 |
| MVT | 1.017 | 1.011 | | | | |
The programs under 0.1 s (2DCONV, ATAX, BICG, GEMVER, GESUMMV, MVT) time start-up and swing as far between copies of A as W does; none of them is claimed. `X4` is a kernel of 16 K threads that load a `float4` from 64 random 16-byte slots each (variant 0) and from random 4-byte-aligned addresses (variant 1): the sim's output hash is the same on A and W for gfx942 and gfx1030, and equals the real RX 6800's (`amd/rx6800` and the card: `d5a8901c3ed4a983` / `8813586ecd033783`).

`bench-amd` canary (mi300x, 44 rows, one thread, 15 runs, cpu 13): counters identical in all 44 rows (and `--check-counters`: 66 workload/target pairs match). W/A, with the A/A of a copy in brackets: `mfma` 1.016 (1.004), `smfmac` 1.012 (1.001), `divergent` 1.018 (1.021), `mem` 0.999 (1.000), `flat` 1.048 (0.989), `smem` 1.030 (0.947), `narrow` 1.019 (1.008), `mfma_fp8` 1.044 (1.000), `p_i64` 1.037 (1.004), `valu_int` 0.988 (0.984), `atomics` 0.989 (0.993); no row below 0.98. The -3 % of `mfma` / `smfmac` and the -2 to -4 % of `divergent` that Round sp17 measured against the round before are not in these numbers against `sp17-speed`, and rows that swing between two copies of the same library (`smem`, `p_sbuf` 1.070, `p_smem_wide` 1.044, `launch`) are not claimed either way. Not recovered by design: that loss is the placement of `gcn_exec.cpp`, which a change anywhere in that unit re-rolls.

### The layout lottery, again
The first version of `wave_load4` was a member defined in the middle of the memory path: CORR 1.04 to 1.06, COVAR 1.03 at equal counters (other rows 2 to 3 % faster). Defining it after the class (the `noinline` stays on the declaration) took CORR and COVAR back to 0.998 / 0.984 and moved the rest within the A/A. The `fix_nan32` change is in `gcn_tier.cpp`, a unit outside link-time optimization, and moved no other row (checked alone: CORR 0.994, GEMM 1.000). `lint_hot_unit_layout` passes.

### Tests (CMake-wired, existing) and mutations
No new test file: the existing same-bits tests already drive both paths, so the evidence is the mutants they kill (each mutant built and run alone against the whole unit).
* `test_amd_gcn_tier_same_bits` (flavour 2: floats with the specials and NaNs, every `v_*_f32` shape of the five corpora, tier on against tier off). Killed: the add, mul, subrev and fma operand order flipped, the packed fma order flipped, the NaN not quieted, the second operand taken after the third, only the first operand looked at, a 32-lane pass over a 64-lane wave. Survived, equivalent: the third operand ignored (the lane then takes the scalar rule, which gives the same bits) and the scalar fallback never run (the vector result of `inf - inf` is the scalar rule's, the default NaN).
* `test_amd_gcn_gather_same_bits` (flavours 21 to 24 hold `global_load_dwordx4` / `flat_load_dwordx4` lanes on every boundary, with lanes on one address and one lane off). Killed: the last word at +8, the pair halves swapped, the word and the last word swapped, a batch of 4-byte loads one short. Survived, equivalent: the high pair at +4 and the middle pair at +8 (the batch declines on the misaligned 8-byte load and the lanes go one at a time, correctly), the same-address rule dropped (loads, single thread), `conc()` not tested (the test runs one thread), the alignment test of the addresses dropped (the batch declines itself). Also: `test_amd_gcn_same_bits`, `test_amd_gcn_memory_same_bits`, `lint_hot_unit_layout` and `bench-amd --check-counters` pass.
* The substring filter `VGPU_GATHER_ONLY=dwordx4` aborts with a double free on `sp17-speed` too (`load_dwordx4` and the other filters pass); not looked into.

### Not done
* `run_groups` (above), and a gfx1030/gfx942 real-card A/B of quad stores. `dwordx4` stores that are not runs still go lane by lane.
* The AMD rows are measured on Zen 5 only; Zen 4 was not part of this round.

## Round sp17: the AMD memory path on real programs (branch `sp17-amd`, base `sp15-speed-pr` 4a5185a3)

The question: the sp16 profile of the real programs put a quarter of the AMD cycles in the memory path (`wave_scalars`, `tier_global`, `memory_run`, `load_batch_in`, `batch_addresses_ok`; 78 % of SYR2K) and 12 % in the `run_groups` loop. Measured first, with `perf record -e cycles:u` on the 20 PolyBench/GPU programs hipified and built for gfx942 (`hipcc -O2`, one thread, `amd/mi300x`), then changed in the order the profile gave.

### What was found
* `MemoryManager::load_batch` has a one-pass path (tests and loads together, Round sp10) that is taken for at most **32** lanes: a CDNA wave is 64, so every gathering load of every wave64 program took the two-step path (`batch_addresses_ok`, then `load_batch_in`): 28 + 8 % of SYR2K.
* `wave_scalars` copied the active lanes' addresses into a list lane by lane (`ctz` walk), gathered the store data and scattered the loaded values through a lane-index list, for a wave that is almost always whole. `tier_global` built its address row the same way.
* The 8-byte loads of a stencil (the compiler merges `A[j-1]` and `A[j]` into one `global_load_dwordx2`, and three neighbours into a `dwordx3`) are on a 4-byte boundary in every other lane: the batch declines on the first misaligned address and **every lane went through `load_words` and `MemoryManager::load_scalar` one at a time** (2DCONV 45 % of its cycles, JACOBI1D 37 %).
* `memory_run` asked every gathering instruction's first lane whether a run starts there, with a call; and its run length was a lane-by-lane loop of about 11 instructions a lane (70 % of the function's cycles).

### What changed
1. `load_batch`: the one-pass path for up to 64 lanes (one constant).
2. `Machine::wave_scalars`: a wave whose lanes from `from` to the last are all active (the usual case) uses the caller's address row and the register rows directly, with counted loops; `tier_global` fills its address row the same way (`amd/src/gcn_exec.cpp`, `amd/src/gcn_tier_machine.inc`).
3. `Machine::load_wave`, `Machine::wave_load3`: the lane-by-lane rule of `load_words` (a pair on an 8-byte boundary is one 8-byte access, a pair on a 4-byte boundary is two word accesses, a three-word load is an aligned pair and a word) done as two batches, one of 8-byte loads and one of 4-byte loads, when the whole-wave batch declines for alignment. Loads only: stores keep their lane order.
4. `tier_global` does not call `memory_run` for a lane whose next lane is off or has the wrong address (the call would count one lane and hand back 0).
5. `include/vgpu/exec/amd_run_scan.hpp`: the run length of the lanes' addresses four lanes to a vector (SSE2) where the low words cannot wrap round 2^32 over the lanes asked about, the lane loop (`run_scan_reference`) otherwise; `memory_run` uses it.

### Results (counters, registers and memory identical in every run)
Zen 5 (server2, `taskset` on one core, `VGPU_THREADS=1`, user-time median of 9 runs (SYRK, SYR2K, LU: 3), A = `sp15-speed-pr`, W = this branch, A2 = a copy of A made at the same time; the box also runs a CI runner, so load was 2 to 13):
| program | W/A | A2/A | | program | W/A | A2/A |
|---|---:|---:|---|---|---:|---:|
| 2DCONV | 0.626 | 0.981 | | GEMM | 0.775 | 1.020 |
| JACOBI1D | 0.673 | 1.000 | | 3MM | 0.767 | 0.977 |
| JACOBI2D | 0.762 | 1.000 | | GEMVER | 0.791 | 1.030 |
| GESUMMV | 0.694 | 1.028 | | ATAX | 0.805 | 1.039 |
| SYRK | 0.675 | 0.960 | | MVT | 0.817 | 1.012 |
| SYR2K | 0.684 | 0.995 | | LU | 0.898 | 1.013 |
| BICG | 0.759 | 1.034 | | 3DCONV | 0.951 | 1.000 |
Zen 4 (server3, `perf stat -e instructions:u,cycles:u`, one run each, pinned): instructions W/A and cycles W/A: 2DCONV 0.515 / 0.59, JACOBI1D 0.557 / 0.63, JACOBI2D 0.678 / 0.73, GESUMMV 0.652 / 0.71, BICG 0.676 / 0.69, ATAX 0.754 / 0.75, MVT 0.755 / 0.78, GEMVER 0.760 / 0.78, GEMM 0.767 / 0.84, 3MM 0.759 / 0.77, SYRK 0.684 / 0.64, SYR2K 0.654 / 0.62, CORR 0.772 / 0.86, LU 0.890 / 0.89, 3DCONV 0.944 / 0.95.
Instructions retired fell 5 to 49 % on all fifteen.

`bench-amd` canary (mi300x, 44 rows, one thread, 9 runs, Zen 5; counters identical in all 44): `mem` 1.295, `narrow` 1.612, `lds` 1.141, `ldsx` 1.070, `flat` 1.051, `buf` 1.045, `launch_wide` 1.048, `launch_mid` 1.043 against an A/A (a copy of A, same window) of 1.012, 1.012, 1.016, 1.007, 0.998, 0.998, 1.000, 1.001. The rows that lose at equal counters, from three windows (the first two with a copy pair): `mfma` 0.971 / 0.969 / 0.965 (the copy pair 0.960; A/A 1.002), `smfmac` 0.970 / 0.981 / 0.977 (A/A 1.001), `divergent` 0.968 / 0.979 / 0.964 (A/A 0.997), `p_sbuf` 0.943 / 0.942 / 0.943 (the copy pair 1.038: it depends on the file, as in Round sp15). Zen 4 (server3, cpu 13, 9 runs, A/A with a copy of A): `mem` 1.239, `narrow` 1.599, `lds` 1.104, `flat` 1.078, `buf` 1.072, `ldsx` 1.052, `smfmac` 1.074 (A/A 0.992), `atomics` 1.031, `launch_mid` 1.024, `mfma_f64` 1.031; no row under 0.97 except `p_sbuf` 0.970 (A/A 1.022) and `apicheap` 0.934 (A/A 1.072, the copy pair swings as far), so none of the Zen 5 losses shows there. Rows that swing between copies of one library whatever the change (`valu_int` 1.06, `mfma_f64` 1.08, `mfma_f64_4b` 1.13, `wave` 1.07 on A/A as well) are not claimed.

### The layout lottery, again
The interpreter is the one AMD unit that is still link-time optimized, so every edit of `gcn_exec.cpp` re-rolls the placement of everything in it. Variants tried on the same rows (all at equal counters): `load_wave` out of line with the run check in `tier_global`: `mfma` 1.00 to 1.01, `smfmac` 0.99 to 1.00, `divergent` **0.92 to 0.93**, `p_sbuf` 0.94; `load_wave` out of line without the run check: `mfma` 0.98, `smfmac` 0.986, `divergent` 0.98 to 0.996, `p_sbuf` 0.926. The run check is worth about 3 % of the instructions and cycles on the gather programs (ATAX 14.01 G -> 13.67 G instructions, SYR2K 316.4 G -> 307.3 G), and the swings of the rows above are the size of the lottery, so the version with both is kept. `lint_hot_unit_layout` passes; the hot units' machine code is unchanged (compared with `objdump -d`), `memory.cpp` differs in one immediate (64 for 32).

### Tests (CMake-wired) and mutations
* `test_amd_gcn_gather_same_bits` (`amd/tests/unit/test_gcn_gather_same_bits.cpp`; new flavours 21 to 24 of `amd::digest_instruction`): every flat / global instruction of the ISA corpora (gfx942, gfx950, gfx90a, gfx1100 and gfx1030, wave32 and wave64) on a wave whose addresses are consecutive, scattered, a run and then a gather (every lane the gather can start at), a word apart for 2 and 3 words (overlap), with lanes on one address and lanes off, run four ways (tier and the whole-wave memory calls on and off, `amd::set_fast_paths`) and held to the lane-by-lane run (tier off, whole-wave calls off). No recorded digest. Mutants killed: gather count one short, the saddr address loop without the offset (or one lane late), the non-saddr address plus 4, a gapped wave taken for a whole one (in `wave_scalars`), store data from the wrong lane, a pair's high word dropped, pair halves swapped (in `load_wave`), the second word of a split pair at +8, the three-word order and the word address of `wave_load3`, three-word stores batched as loads (segfault). Not killable, so equivalent: the one-read-per-address copy (no other thread writes in the test), a wrong high shift in the address row (the batch declines and the lanes go one at a time, correctly), the whole-row loop accepting a gap (it only computes addresses nothing uses), the alignment test of `wave_load3` (a misaligned pair makes the batch decline), the last aligned pair copied in `load_wave` (the declined batch had already left the aligned lanes' values).
* `test_gather_sp10` (existing): counts 1 to 64 for the one-pass path, 65 to 104 for the two-step one, the wrong lane anywhere in the wave. Mutant killed: the bounds test skipped for lanes past 32.
* `test_amd_run_scan` (`tests/unit/test_amd_run_scan.cpp`): 200,000 random rows against `run_scan_reference` (pair and base forms, strides 4, 8, 16, runs broken at any lane, a high word that differs, a wrapping low word, every reach). Mutants killed: the tail off by one, the high word expected zero, the wrap check dropped, the step wrong, the first mismatching lane off by one; the base form's redundant range test removed as equivalent.
* The existing `test_amd_gcn_memory_same_bits`, `test_amd_gcn_tier_same_bits` (and `_simd_base`), `test_amd_gcn_same_bits`, `test_amd_matrix_fast`, `test_amd_gcn_memruns` (five targets), `bench_amd_counters`, `lint_hot_unit_layout` and all 284 `amd` tests (4 fail on `sp15-speed-pr` too: they need targets this configuration does not build) pass.

### Card oracle (server2: RX 6800 = HIP device 0 gfx1030, RX 6700 XT = device 1 with `HSA_OVERRIDE_GFX_VERSION=10.3.0`)
`gather_oracle.hip` (kept out of the repo): 12 integer-only kernels (overlapping `dwordx2/x3/x4` loads at word alignment, gathers, a pair gather, a run then a gather, scatters, pair stores, byte and short accesses, lanes on one address, partial waves) with one FNV hash per output buffer. The hashes on the sim equal the card's on both cards (`amd/rx6800` and `amd/rx6700xt`), A and W identical on `amd/rx6800`, `amd/rx6700xt` and `amd/mi300x` (and the gfx942 hashes equal the gfx1030 card's), and the same with the default thread count.

### Not done
* `dwordx4` loads on a 4-byte boundary (a quad on a 16-byte boundary is one atomic access, the others pairs and words) and the stores of the same shapes still go lane by lane when they are not runs.
* The `run_groups` loop (12 % of the AMD cycles): about 45 instructions of bookkeeping around each handler call (four read-modify-write counters, the state checks of `tier_step`, the `Inst::next` chain), spread out; nothing small and bit-safe stood out, and the `divergent` row says what an edit near it costs in layout.
* `fix_nan32` (25 % of GRAMSCHM, 13 % of LU, programs that compute with NaNs): `std::fma` is called for lanes whose operands are NaN already.

## Round sp16: BSYNC without the store-forwarding stall (branch `sp16-bsync`, base `sp15-speed-pr` 138da21f)

### What the profile said (server3, Zen 4, `perf record -e cycles:u -c 100000`, SASS `divergent`, one thread, base build)
`Runner::step_warp` is 43 % of the row, `pick_group_avx2` 16 %, `blend_m_k` 11 %. Inside `step_warp` three instructions carry 6.0 % + 6.0 % + 4.6 % of the function's samples (about 7 % of the row): the `movb`/`mov` stores of `wait()` (`wait_kind[l]` a byte and `wait_arg[l]` a word, one pair for every lane of the group, in a loop whose length is the lane count) and, a few instructions later, the first 16-byte loads of `waiting_on()` (`pcmpeqb` on `wait_kind`, `pcmpeqd` on `wait_arg`) that read the same arrays back at the same BSYNC to see whether the barrier is complete. A load that spans stores still in the store buffer cannot be forwarded and waits for them to retire. That is the stall the previous round measured; it is confirmed. After the change those samples are gone and the remaining profile of `step_warp` is flat (no instruction above 3.3 % of the function).

### What changed
1. `sass_lanes.hpp` `wait()`: a partial set of lanes is entered with vector blends (the 32 kind bytes as two 16-byte load/blend/store, the 32 argument words as eight), not a store a lane. The loop had a length no predictor learns in a diverged warp, and its stores were what the read-back stalled on. A hybrid that keeps the store loop for 1 to 3 lanes was measured and is worse (cycles 0.990 against 0.971 of the baseline for the always-blend version, and more instructions): not kept.
2. `sass_lanes.hpp` `bsync_arrive()`, used by the tier's BSYNC (`tier_bsync`): wait and release in one step. When some member of the barrier is neither waiting nor among the arriving lanes (the first and middle arrivals) the barrier cannot complete and nothing is read; otherwise the lanes that already waited are read before the new ones are entered and the arriving lanes are added from the mask, so the arrays that were just written are not read back. Alone with the old store loop it gave 0.984 of the baseline's cycles; on top of the blend it is worth about 0.3 % (0.973 against 0.976 without it, inside the noise of one box), kept because it is also what removes the reads when the blend is not taken (the non-SSE2 build).

The interpreter fallback (`exec_control`, `release_bsync`) is unchanged: the tier handles every BSYNC the benchmarks execute. The PTX interpreter and the AMD engine do not include `sass_lanes.hpp`.

### Results (same bits: outputs identical, counters identical in every row of every window, 14 workloads x SASS and PTX)
Instructions retired, Zen 4, `divergent` SASS: 3,403.67 M -> 3,405.20 M (+0.05 %: the blend costs a few instructions where the loop was short). Cycles (`perf stat` user cycles, cpu 5, 9 alternating runs, medians, load 30 to 45 from other sessions on the shared box): 798.4 M -> 775.7 M, 0.972; A/A of two copies of the base 1.000.

`bench.py --ab` (copies of each build's libraries made fresh in the same minute: see the caveat below). Zen 5 (server2, cpu 9, 15 runs), two kinds of window:
* all 28 rows, A/B and then A against a second copy of A: `divergent` SASS 1.099 (A/A 1.059), `mem` SASS 1.055 / 1.041 in two windows (A/A not repeated), `launch` SASS 0.975 / 0.969 in the A/A, `int_alu` SASS 0.982, `atomics` SASS 0.986, PTX `wmma` 0.975 in the A/B window and 1.005 in the A/A. All other rows 0.99 to 1.02 in both windows. PTX rows 1.00 +-0.01 as expected (no PTX code changed).
* three fresh copies of each side, paired (A_i vs B_i), SASS rows only, with A/A pairs (A_a vs A_b, A_b vs A_c) and B/B pairs: `divergent` 1.036, 1.037, 1.038 (A/A 1.001, 0.996; B/B 0.997, 0.996); `mem` 1.018, 1.018, 1.013 (A/A 0.989, 1.002; B/B 0.996, 0.996); `local` 1.000, 1.004, 1.001; `f32_fma` 1.019, 1.015, 1.010 (A/A 0.983, 0.995); `launch` 0.993, 1.016, 1.017 (A/A 1.005, 0.986); `atomics` 0.985, 1.067, 0.972 (A/A 0.922, 1.086; B/B 0.999, 0.989: the copies of A differ among themselves by up to 9 % on this row, a layout lottery, so no claim).
So `divergent` +3.7 % on Zen 5 against an A/A of 1.00, and the first window's 1.099 is half the copy lottery. `mem` +1.3 to +1.8 % is above the A/A spread of the same copies but is code that did not change; do not count it.

Zen 4 (server3, cpu 5, 9 runs, 28 rows, A/B and A/A): `divergent` SASS 1.029 (A/A 0.991), consistent with the 0.972 cycles above; every other row 0.99 to 1.03 with the A/A inside the same range (`f64_subnormal` SASS 1.027 against an A/A of 1.007, `mem` SASS 1.021 / 1.005, both with best-of-run ratios of 1.00).

Real programs that diverge (Zen 4, `perf stat` instructions and cycles, cpu 5, medians of 5 alternating runs, outputs identical except particlefilter, which differs from run to run in the base too), B/A cycles, instructions in brackets: needle 0.982 (1.011), pathfinder 0.986 (1.001), particlefilter_float 0.995 (0.997), sc_gpu 1.011 (1.000), srad_v2 0.975, nn 0.978, sortingNetworks 1.008 (1.001), mergeSort 1.019 (0.999), radixSortThrust 0.977 (0.999), segmentationTreeThrust 0.985 (1.000), interval 0.994 (0.996), shfl_scan 0.997, histogram 1.009, reduction 1.015 (1.000); the programs under 0.1 G cycles (bfs 1.019, nn, MC_EstimatePiP 1.047) time process start. The A/A of the same harness (two copies of the base) reads needle 0.995, pathfinder 0.975, sc_gpu 0.997, sortingNetworks 1.000, mergeSort 1.012, histogram 1.001, radixSortThrust 0.999. So the real programs gain nothing measurable: their cycles move by the A/A floor (+-2 %). That is expected, because they run BSYNC rarely compared with the arithmetic (`divergent` is 22 host instructions per warp instruction; Rodinia and the samples are dominated by loads, shared memory and the tensor kernels). needle's +1.1 % in instructions is the blend (its groups are small).

### Caveat found on the way: copies of one library differ
Two builds with byte-identical `libcudart.so` can differ by 5 to 100 % on a row depending on the file's age: with the base library as built an hour earlier, PTX rows read 1.3x to 2.2x slower than the same file freshly copied (every PTX row, in both the A/B and the A/A window, on Zen 5; reproduced with `time`), and the SASS rows of the fresh copies spread by up to 9 % (`atomics`). The likeliest cause is how the page cache holds the text (large folios for a file just written, small ones for one read back after readahead; a 4.5 MB text on a 4 KB iTLB). Use copies written at the same time for every side of an A/B, as this round did, and an A/A over several copies, not one.

### Tests (CMake-wired, no GPU) and mutations
* `test_sass_lanes`: `bsync_arrive_agrees_with_wait_then_release` against `wait` then a lane-by-lane `waiting_on` and the completeness rule, over 60,000 random warp states (members none, random, or exactly the waiting plus arriving lanes with and without one lane missing; other kinds and barriers waiting) and the whole state compared with `memcmp`; it counts that the release, hold and no-member cases each occur more than 2,000 times. The existing `wait_state_helpers_agree_with_the_lane_loop` covers the blended `wait` for random masks.
* Mutants, each built and run on server2, all killed: the cannot-complete shortcut ignoring the waiting lanes; the arriving lanes not added; no limit to the members; a release without completeness; an empty release counted as complete; the wait not entered; the kind blend shifted by one lane; the second half of the kinds written to the first; the argument blend shifted by one lane; the old words dropped by the blend (arguments, kinds); `waiting` not updated by the blended path.
* Run on server2 in the final tree: `lint_hot_unit_layout` (also on server3), `test_tier`, `test_sass_lanes`, `test_sass_alu_runs`, `test_sass_atomics`, `test_sass_block_reuse`, `test_sass_dispatch`, `test_sass_lockfree`, `test_sass_mem_rows`, `test_sched`, `test_exec`, `test_exec2`, `test_fastpath`: all passing.

### Not done
* `step_warp`'s remaining bookkeeping (the pc writes at a branch, `popc(group)`, the counters) is spread flat over the function in the divergent profile; no single edit worth the layout re-roll was found. `wake()` of a partial set is still a store a lane (it is rare: a barrier releases whole warps once its groups have met).
* `pick_group_avx2` (11 to 16 % of `divergent`) reads the 32 program counters as sixteen 16-byte halves with `vinserti128`; not looked at further.
* AMD rows were not re-run: `libvgpuhip` does not include the changed header.

## Round sp16: a warp's load of one address and a warp's store on a stride (branch `sp16-sass`, base 705ed36f = main with #460)

Where the time goes now (server3, `perf record` cycles:u, one thread, A = main at 705ed36f, the 66 corpus programs of tools/bench/real: PolyBench, Rodinia, HeCBench). Mean share over all programs: `step_warp` 12.0% (sp15-step's), `MemoryManager::load_batch` 5.3% (up to 38% in SYR2K, 36% in SYRK, 27% in ATAX), `warp_rows_k` 5.0% (24% in GRAMSCHM), `h_global<false>` 2.6%, `load_run` 2.3%, `exec_mem` 1.8% (the atomics rows), `h_shared<false>` 1.6%, `init_block` 1.5% (12% in rodinia gaussian, 17% in HeCBench blockAccess), `imad_wide` 1.4%. By instructions retired the gather-bound PolyBench rows are `load_batch` 35% + `count_global_progression` 13% + `warp_rows_k` 12% + `h_global` 11% (SYR2K). IPC there is 4.5 to 5.2, so only fewer instructions help, except that the gathers themselves wait on L1 misses (`ex_no_retire.load_not_complete` is 27% of the cycles of SYRK).

### Part 1: a load of one address is one load (`MemoryManager::load_broadcast`)
Every lane of the warp loads the same address (a loop-invariant, `A[i][k]` of a matrix product, a kernel argument read through a pointer): `warp_rows_k` already reports "on a stride" and the stride (0). `h_global` now asks `load_broadcast(a0, 32, bytes, fv)`: the checks of `load_batch` (no fault armed, no host mapping in reach, aligned, inside the allocation the first lane is in), one chunk lookup, one load (an untouched chunk reads 0, a uniform chunk its byte repeated, as `load_scalar` does), the word copied to the 32 entries of `fv`. The counters are counted as before (`count_global_progression`, stride 0). A word that another host thread stores to between the lanes' loads could be seen differently by two lanes before; now all lanes see one value, which is a value the one-by-one loads could have given.

### Part 2: a store on a stride, a chunk at a time (`MemoryManager::store_strided`)
A full warp's store on a stride of up to half a chunk either way (`warp_rows_k`: `ri.progression`, `ri.stride`): the lanes in one real chunk are plain stores after one lookup; a lane in an untouched or uniform chunk is `store_batch`'s (a uniform chunk is left alone by a store of its own byte, any other is materialized; a failure to get memory is thrown with `*failed_at` the lane, as there). Checked at the two end lanes (the lanes between are between them). Everything else, and every decline, is `store_batch` unchanged.

### Results (server3, one thread, cpu 10, `perf stat` instructions:u and cycles:u, alternating A B, A = 705ed36f built here and frozen, B = the branch; Mann-Whitney p on cycles; counters and stdout identical in every row)
| program | runs | instructions B/A | cycles B/A | p |
|---|---:|---:|---:|---:|
| polybench ATAX | 15 | 0.812 | 0.851 | <0.001 |
| polybench BICG | 15 | 0.818 | 0.890 | <0.001 |
| polybench GESUMMV | 15 | 0.863 | 0.917 | <0.001 |
| polybench MVT | 15 | 0.810 | 0.862 | <0.001 |
| polybench GEMM | 15 | 0.818 | 0.893 | <0.001 |
| polybench SYRK | 7 | 0.833 | 0.887 | 0.002 |
| polybench 2MM | 7 | 0.811 | 0.929 | 0.002 |
| polybench 3MM | 7 | 0.806 | 0.891 | 0.002 |
| polybench CORR | 7 | 0.891 | 0.943 | 0.002 |
| polybench COVAR | 7 | 0.891 | 0.941 | 0.002 |
| polybench GRAMSCHM | 7 | 0.838 | 0.959 | 0.025 |
| rodinia hotspot (no change expected) | 15 | 1.000 | 0.998 | 0.44 |
| rodinia gaussian | 15 | 1.002 | 0.998 | 0.19 |
| polybench 2DCONV | 15 | 1.001 | 1.004 | 0.15 |
| rodinia needle | 15 | 1.000 | 0.987 | <0.001 |
| rodinia backprop | 15 | 1.000 | 0.994 | 0.011 |

A/A (A against itself, 15 runs): ATAX 0.998, GEMM 0.999, hotspot 0.999, gaussian 1.000. The broadcast alone (the store path switched off) gives ATAX 0.848, GEMM 0.887, CORR 0.999; the store path alone is what moves CORR and COVAR (0.94). bench.py canaries (13 workloads x SASS and PTX, 15 runs, cpu 10, load 5 to 7 from other jobs, so wide spreads): counters `identical` in all 30 rows, every row between 0.97 and 1.03 and inside its own spread, except `texture` (SASS 1.032x, PTX 1.028x, p 0.001; 1.036x and 1.028x in an earlier run: the code is not on that path, so placement), `divergent` SASS 0.988x (p<0.001, spread 8%).

### Tests (CMake)
* `test_warp_stride_access` (new): `load_broadcast` against `load_scalar` for 1 to 32 lanes at widths 1/2/4/8, over untouched, uniform and real chunks, at chunk edges, the last word of the allocation, one past it, below the base, misaligned, and its declines (more than a warp, no lanes, a width that is none, no allocation); `store_strided` against 32 `store_scalar` calls on an identical world (the bytes of the whole allocation and `resident_bytes()`, so that a uniform chunk left alone and a chunk made real are told apart), strides of either sign from 0 to half a chunk and one element past it (declined), values that are a uniform chunk's own byte; and the edges deterministically: a lane exactly on the first byte of the next chunk (both directions, widths 1 to 8, strides of 1 to 1000 elements), a last lane ending exactly at the end of the allocation (taken) and one byte past it (declined), the first lane at the base or one byte below it.
* `test_sass_generic_mem`: mode 4 (`global_accesses_on_a_descending_stride`: lane l at base + (31 - l) * S, strides 4 to 200: a stride taken as unsigned or lanes counted from the wrong end read the wrong words) and mode 5 (`global_accesses_off_a_stride_over_a_full_warp`: the stride doubles at lane 16, a full warp in global memory that is not on one stride: it must go to the batch), tier against the interpreter and the written-out rule: bytes, registers, every counter. Both also under `VGPU_SIMD=base` (no row kernel: the paths are off).
* Mutations, each rebuilt and run, each fails a test: store last-lane bound `>=` to `>` (the edges test); store segment ends one lane late `io <= kChunkSize` (edges); the broadcast of a uniform chunk gives 0; the store always materializes; the broadcast's alignment test removed; the tier broadcasts for every progression (generic_mem: stride tests); the store's base moved by one element; the store takes any full warp, not only progressions (mode 5); the store's stride negated (descending).
* Also run, all pass: `test_gather_sp10`, `test_memhot_sp8`, `test_memfast`, `test_sass_mem_rows`, `test_tier`, `test_memcopy_direct`, `test_sass_atomics`, `test_sass_access_count`, `test_memrmw`, `test_memalloc`, `test_sass_block_reuse`, `test_sass_ldgsts`, `test_sass_dispatch`, `test_hostmap_cache`. Not run (no nvcc on server3, CI): `e2e_sass_tier`, `e2e_sass_archs`, the AMD guards, ASan/UBSan/TSan builds (the change adds no atomics beyond `load_at`/`store_at`, which are the manager's own).

### Not kept, with numbers
* **A strided gather load** (`load_strided`: the chunk looked up once for the lanes that share it, the lanes in it plain loads). Instructions down 17 to 24% in every gather row; cycles ATAX 0.89, MVT 0.90 (better than the broadcast alone by 2 points), but **SYRK 1.07 to 1.09 (stride 4096 bytes, 16 lanes to a chunk)** with 24% fewer instructions: `ex_no_retire.load_not_complete` 8.7 G to 14.6 G cycles, the hardware prefetcher's allocations halved (0.42 G to 0.22 G), L1 misses down (1.40 G to 1.21 G). The cost is in waiting for the loads, not in work. Not the loop exit (branch-misses 5.9 M both), not store forwarding (writing the words straight into the register row: same), not the stores' 4 KB aliasing, not a software prefetch of the next line (worse), not the length of a run of tight loads (a break every 4 lanes: same), not a dependency every 8 lanes (same or worse). Dropped: the broadcast (stride 0) alone is a win on every row it touches, SYRK 0.887.
* **Transparent huge pages for the chunks** (`GLIBC_TUNABLES=glibc.malloc.hugetlb=1`, an experiment, not a change): dTLB load misses 3.8 M to 11 K in ATAX, cycles 0.975; SYR2K cycles 1.00. The gathers are L1-miss bound, not TLB bound.
* **Colouring the chunks** (each chunk's base offset by a different multiple of 64 bytes, so that the lanes of a stride-4096 gather spread over the L1's sets; experiment with leaked chunks): ATAX L1 misses 190 M to 68 M (-64%), cycles 0.967; MVT 0.97; GESUMMV 0.98. A real gain of 2 to 3% for a change in how every chunk is allocated and freed (the pointer's low bit is a tag, the delete[] needs the original): not worth it alone.

### Not done
* `init_block` is 12% of gaussian and 17% of blockAccess (small blocks, many of them): two `rep stos` of about 3 KB each per warp (`new (&w) Warp()` and `w.r.assign(nregs * 32, 0)`; `ur[256]`, `pc[32]`, `cga_target[32]`, `rpc[32]` are most of it). A reset of only what a run can have changed needs a test that a reset warp equals a fresh one.
* `warp_rows_k` spills six ymm registers to the stack in the unrolled loop (the precomputed expected-address chain) and is 5 to 15% of the gather rows; a version with `expect` advanced in the loop and no spills was not tried.
* `count_global_progression` is 70 instructions with a six-register prologue and epilogue (13% of SYR2K's instructions); inlining it into `h_global` changes the placement of `h_global`.

## Round sp19: process start, and a PTX module is parsed a kernel at a time from 16 KiB (branch `sp19-start`, base ef41312a)

The question was where the fixed cost of a short program goes (CI, the playground, many small tests). Measured on server3 (Zen 4, `perf stat` instructions:u and task-clock, two CPUs pinned, 30 runs a side rotated over three fresh copies of each library directory; the benchmark program's cached binary for the module cases, two `g++`-built runtime-API probes for the rest).

### Where the start goes (base ef41312a, NVIDIA shim, CUDA 13)

| program | task-clock | user instructions | what it is |
| --- | ---: | ---: | --- |
| `devq` (device count + properties, no module) | 2.1 ms | 2.9 M | 229 page faults, 0 threads, ~35 `openat`, 32 `mmap` |
| `memcpy` (malloc, memcpy, memset, free; no module) | 2.9 ms | 4.1 M | |
| benchmark `startup`, SASS (default), verdict cache warm | 3.2 ms | 4.4 M | context, module registration, one launch |
| the same with `VGPU_SASS_CACHE=0`, or a cold cache | +0.7 ms | 9.1 M | decoding and checking the module's SASS; the cache saves it from the second run on |
| the same on the PTX engine (`VGPU_SASS=0`) | 18.4 ms | 76.9 M | parsing the module's PTX: 81 KB, every kernel, to launch one |
| HIP `startup` (gfx942 shim) | 3.5 to 4 ms | 6.2 M user | one worker thread; 703 page faults; most of the time is the kernel's (exec and faults) |

What the time is, for a program with no module (`devq`, 2.1 ms): the dynamic loader. 77% of the user-mode samples of 300 runs (`perf record -F 20000 -e cycles:u`) are in `ld-linux`, and the simulator library itself is 6%. `LD_DEBUG=statistics` reports a smaller figure, 512 k cycles in the loader before `main` (258 k relocating 2338 symbols, 170 k loading objects); the two were not reconciled (the samples include the lazy symbol resolution after `main` starts). `LD_DEBUG=bindings`: 2503 symbol bindings in all, of which 1729 are libstdc++'s own, 551 are the shim's own exported symbols, 47 are the shim's references to libstdc++ and 36 to libc. There is no thread, no large allocation and no environment scan; `strace -c` shows 0.6 ms of syscalls under tracing. `LD_BIND_NOW=1` makes it worse (8.3 M cycles against 6.0 M), so lazy binding is the right setting. The first two lines of the table are what a program that touches no module costs; they have no hotspot (the earlier profile of this document, "a 3 ms program ... none above 1.5%", still holds).

Not done, with the reason:
* `-Wl,-Bsymbolic-functions` (or hidden visibility) would remove the 551 self-bindings (22% of the lookups). It is not safe: `libcuda.so.1` and `libcudart.so.N` each carry the simulator's core, and the two copies become one machine because the dynamic linker lets the first library's definitions interpose on the second's (`runtime::owner()` in `src/runtime/shared_runtime.cpp` looks the copy up by name for the libraries loaded privately). A symbolic link would give each library its own machine. The saving is about 0.1 ms of 2.
* A static libstdc++ in the shims would remove 1729 of the 2503 bindings, but puts a second C++ runtime in every process that has its own. Not tried.
* The HIP start is user 1.3 to 1.7 ms and system 2.1 to 2.5 ms on four CPUs (the code object's msgpack parse and the decoder's `table()` are 5% and 3% of it by a frame-pointer call graph, which is not reliable here). Nothing is above the 2% bar.

### What changed

`kLazyModuleBytes` goes from 2 MiB to 16 KiB (`src/runtime/runtime.cpp`). A module of 16 KiB or more is split into its kernels and each is parsed when first looked up, as a module over 2 MiB already was. The PTX parser costs about 0.9 k host instructions a byte, so an 81 KB module spent 14 ms parsing all its kernels to launch one. `VGPU_LAZY_MODULE_BYTES` and `VGPU_PARSE_WHOLE=1` still move or turn off the threshold. 16 KiB, not 0: a smaller module parses in under 3 ms, and keeps the whole parse's behaviour of refusing a kernel the simulator cannot parse when the module loads (`unsupported_ptx_error_names_profile` needs it, and fails with the threshold at 0).

The one behaviour that moves: a module between 16 KiB and 2 MiB with an unsupported instruction in a kernel nothing launches now loads, and the error is raised when that kernel is looked up (it already was so above 2 MiB). The nightly "Lazy modules" workflow runs the whole e2e suite with every module lazy and has passed daily since it was added.

The by-opcode line of the `VGPU_COUNTERS` report ordered equal counts by opcode id, and ids are handed out in the order the parser first meets a mnemonic, so a lazily parsed module printed `cvta=1 mov=1 mul=1` where the whole parse printed `mov=1 mul=1 cvta=1`. The line is now built by `runtime::opcode_summary`, equal counts ordered by name: the same text whichever way the module was parsed. This changes the text of existing reports where counts tie (on the PTX engine; the SASS engine's line is unchanged); the numbers do not change.

### Results

| case | instructions B/A | task-clock A | task-clock B | saved | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| PTX engine, benchmark `startup` (81 KB module) | 0.084 | 18.4 ms | 3.3 ms | 15.1 ms | <0.001 |
| PTX engine, `mem quick` | 0.833 | 61.6 ms | 46.6 ms | 15.0 ms | <0.001 |
| PTX engine, `divergent quick` | 0.762 | 39.6 ms | 25.4 ms | 14.2 ms | <0.001 |
| PTX engine, `int_alu quick` | 0.856 | 54.2 ms | 39.4 ms | 14.8 ms | <0.001 |
| A/A, PTX `startup` (same library, other copies) | 1.000 | 18.4 ms | 18.4 ms | 0.03 ms | 0.78 |
| PTX engine, `api quick` (does not load the module) | 1.000 | 11.7 ms | 11.7 ms | 0.03 ms | 0.31 |
| SASS engine, `startup` | 1.000 | 3.18 ms | 3.18 ms | 0 | 0.94 |
| SASS engine, `mem quick` | 1.000 | 38.2 ms | 37.8 ms | 0.4 ms | 0.006 |
| `devq` | 1.000 | 2.12 ms | 2.12 ms | 0 | 0.18 |
| `memcpy` | 1.000 | 2.87 ms | 2.85 ms | 0.02 ms | 0.20 |

The SASS engine and programs with no module do not change (the `mem` row is 1%, below the 2% bar, and instructions are equal). Counters and stdout, all 16 workloads on both engines (`VGPU_COUNTERS=1`): the branch with the default threshold and with `VGPU_PARSE_WHOLE=1` print the same text for every workload except `atomics`, which differs from itself between two runs (host-atomic order); against the base the numbers are identical and only the order of tied opcodes in the by-opcode line differs (PTX engine), more lines on `launch` and `mem` where there are many kernels.

### Tests (CMake-wired, `test_runtime`)

* `modules_from_16_KiB_up_parse_kernels_when_first_used`: a 20 KiB module with an unsupported kernel loads, its good kernel runs, looking up the unsupported one gives `UnsupportedPtx`.
* `modules_under_16_KiB_are_parsed_whole`: the same module at 4 KiB is refused at load.
* `opcode_summary_orders_equal_counts_by_name`, `opcode_summary_keeps_ten_and_counts_the_rest`.
* Mutations, each rebuilt and run on server3, each fails a test: threshold back to 2 MiB (the first), threshold 0 (the second and `unsupported_ptx_error_names_profile`), no name tie-break (the third), eleven opcodes shown instead of ten (the fourth).

Not run: the e2e suite (no nvcc on server3); the nightly lazy-modules workflow and the regular CI cover it, and ASan, UBSan and TSan (the change adds no shared state).

## Round sp20 (zen5-check): `sp18-speed` against `sp17-speed` on Zen 5 (docs only, no code changed)

The question: does the combined branch (`sp18-speed` db576493 = SASS global-memory path + sp18-amd + sp19-amd) lose anywhere against `sp17-speed` dd3adb43 on Zen 5? Host server2 (Ryzen 9 9950X), Release, GCC 13.3, `VGPU_THREADS=1`, every run pinned to cpu 13, load 4 to 7 (timed runs wait for load < 8), `nice -n 19`. A and W were built from `git archive` trees in separate directories, then the build directories were copied at the same moment to equal-length paths (`runA`, `runW`) with an A/A copy (`runB`, a third copy of A's bytes). Counters identical in every row of the SASS/PTX canary and the 44-row mi300x canary; outputs identical in every PolyBench and Rodinia program except `particlefilter_float` (DIFFER in A/A as well: the known nondeterministic program).

### SASS and PTX canary (`bench.py --ab`, 13 workloads x 2 engines, 15 runs; W/A median ratio, > 1 means W is faster; A/A in brackets)
| row | SASS | PTX |
|---|---:|---:|
| int_alu | 0.993 (1.001) | 0.997 (0.998) |
| f32_fma | 1.010 (1.012) | 0.999 (1.001) |
| f64_fma | 0.995 (0.990) | 1.002 (0.995) |
| wmma | 1.000 (0.999) | 0.991 (1.002) |
| mma_ptx | 0.997 (0.999) | **1.042** (1.005) |
| mem | **1.043** (1.007) | 1.000 (0.992) |
| shared | **1.035** (1.004) | 1.000 (0.957) |
| atomics | 0.980 (1.013) | 0.995 (1.002) |
| local | 1.003 (0.999) | 0.983 (0.996) |
| divergent | 1.014 (0.997) | 0.996 (0.998) |
| warp | 1.002 (1.011) | 1.002 (1.000) |
| launch | 1.013 (0.999) | 0.997 (0.970) |
| api | 1.007 (1.010) | 1.007 (1.004) |

No row loses 3 %. The two rows under 0.985 (SASS `atomics` 0.980, PTX `local` 0.983) are inside the A/A swing of the window's other rows (PTX `shared` 0.957 and PTX `launch` 0.970 in the A/A; SASS `atomics` read 1.013 in it).

### Real programs (`real.py`, cpu 13, 7 runs, 3 for programs over 30 s; W over A of the median wall time; A/A of the same window: 0.98 to 1.02 in every row)
PolyBench (gather rows first): ATAX 1.16, GEMM 1.13, 3MM 1.12, MVT 1.12, SYR2K 1.09, SYRK 1.09, GESUMMV 1.09, JACOBI1D 1.09, BICG 1.05, 2MM 1.05, CORR 1.03, COVAR 1.04, FDTD-2D 1.04, GRAMSCHM 1.04, LU 1.02, GEMVER 1.04, 3DCONV 1.02, 2DCONV 1.02, JACOBI2D 1.03, ADI 1.03 (geometric mean 1.064x, no program under 1.00). Rodinia: sc_gpu 1.06, particlefilter_float 1.04 (outputs differ in the A/A too), backprop 1.03, needle 1.02, lud 1.01, hotspot / gaussian / lavaMD / srad_v2 1.00, nn 1.01, bfs 1.04 (10 ms, start-up), pathfinder 0.99 (geometric mean 1.013x). The programs under 0.3 s time start-up as much as simulation.

### mi300x AMD canary (`bench-amd`, 44 rows, 15 runs; W/A, A/A in brackets; the rows that moved)
Faster: `mfma_fp8` 1.053 (0.976), `p_smem_wide` 1.048 (0.991), `mfma_bf16` 1.041 (0.990), `mfma` 1.037 (0.989), `flat` 1.033 (0.992), `p_perm` 1.028 (1.023), `divergent` 1.025 (1.009). Four rows read 0.95 to 0.97: `p_lane` **0.946** (A/A 1.004), `p_sdwa` 0.968 (1.007), `smem` 0.963 (0.979), `p_sbuf` 0.970 (0.971); `atomics` 0.975 (0.997). Everything else 0.98 to 1.02. The `mfma` / `smfmac` / `divergent` losses that Round sp17 measured are not present against `sp17-speed`.

A repeat of those rows (21 runs, three windows) gave the same: `p_lane` 0.944 / 0.949 (A/A 1.005), `p_sdwa` 0.969 / 0.971 (0.998), `smem` 0.966 / 0.973 (0.980), `p_sbuf` 0.968 / 0.967 (0.970), `atomics` 0.988 / 0.983 (1.013).

### The cause of `p_lane` and `p_sdwa`: the copy of the library, not the code
`p_lane` is a 15 ms device phase and the 0.9 ms gap is the whole-process gap (user time +0.7 ms, page faults equal at 803 / 804). The profile (SIGPROF sampler, 15 samples) puts the time in the same handlers in A and W, and none of them is touched by `sp18-speed` (`h_int_avx2`, `run_lane`, `h_readlane`; the AMD changes are `fix_nan32` and the dwordx3/x4 memory path). `libamdhip64.so.6.4` of the two differ, but the hot handlers sit at the same 64-byte offsets in both. What decides it is the file copy. Median device CPU time of `p_lane`, 31 interleaved runs, byte-identical copies of A's build and of W's build, all copied at the same moment:

| build directory | bytes of | p_lane device ms | p_sdwa device ms |
|---|---|---:|---:|
| runA | A | 14.8 | 74.0 |
| runB (the A/A) | A | 14.7 | 73.4 |
| runA3 | A | 15.7 | 73.7 |
| runA4 | A | 15.7 | 73.8 |
| runW | W | 15.7 | 76.0 |
| runW3 | W | 15.6 | 74.2 |
| runW4 | W | 15.6 | 74.5 |

Three copies of A's bytes give 14.8 / 14.7 / 15.7 / 15.7, so `p_lane` moves by 6 % between copies of one library; W's copies are all on the slow side. An earlier pair did the same (an unrelated one-function edit to `src/core/error.cpp`, which leaves `libamdhip64` byte-identical in A and in W: the edited A copy read 15.4 ms, the edited W copy 14.7 ms, the reverse of the unedited pair). Page-cache placement of the mapped file (physical addresses of the text pages) is the only thing that differs, so this is Round sp16's "copies of one library differ", and an A/A made with one copy cannot see it: here the A/A copy happened to land with A. `p_sdwa` is the same effect at 1 to 3 % (W copies 74.2 to 76.0 against A copies 73.4 to 74.0). `smem` and `p_sbuf` have the same sign in the A/A (0.979 / 0.971). No code was changed; a fix by layout is not possible for page placement.

### Caveats
* The A/A control is one copy; the p_lane table shows that several copies are needed for a 15 ms row. Rows under 20 ms (`p_lane`, `p_dpp*`, `startup`) are not claimed either way.
* The PolyBench programs here are the MINI datasets scaled by the corpus (0.1 s to 120 s); GEMM-class programs measure at 0.5 s and mostly time the host reference check as well.
* Zen 4 was not part of this round. Timed windows were at load 4 to 7 from other sessions.

## Round sp21z (zero-fill and block setup) (branch `sp21-zero`, base 85ec63a5)

Two overheads from a survey of main: the zero-fill of device-memory chunks (HeCBench `atomicCost`: 7.2 M page faults, 8.8 s of 31 s in the kernel) and the per-block set-up of the SASS executor (`init_block`: rodinia gaussian 12%, HeCBench `blockAccess` 17%). Host: server3 (Ryzen 7 PRO 8840U, Zen 4), Release GCC 13.3 LTO, `VGPU_THREADS=1`, `perf stat` instructions:u and cycles:u (plus page faults, task-clock, user and system time), cpu 12, alternating A B (order flipped each pair), 15 runs (21 for the 4 ms row), three fresh copies of each side's library directory used in rotation, A = main 85ec63a5 built here and frozen, B = this branch. Mann-Whitney p. Standard output with the timing lines dropped identical in every run of both sides. Load 2 to 13 from other sessions on other cpus.

### What the profile said, and a correction
* `atomicCost` is **not** zero-fill churn. It never frees what it touches: its first kernel faults (the card is 12 GB, the second 7.4 GB buffer is refused, and every later call fails on the sticky error), so `cudaFree` never runs. The 7.2 M faults are first touches of 29 GB of live memory: 7.4 GB of chunks, the program's own two 7.4 GB host buffers (a copy into one, a compare reading the other), and the rest. A pool of cleared chunks (up to 20 GB, as an upper bound) changed the fault count by 0.01% and the cycles by +2.1% (the extra clearing, p<0.001). There is no free-then-zero to avoid there.
* The lever for first-touch cost is huge pages: `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` on the unchanged build took faults 7.2 M to 16 K, system time 9.7 s to 1.6 s, wall 35.8 s to 24.0 s (one run each; it also covers the program's own `malloc`, which the simulator cannot touch).
* `init_block` on gaussian (`perf annotate`): the 2.4 KB `rep stos` that zeroes a new `Warp` was 32% of the function, the 392-byte `LaunchStats` temporary 6%, `mbar.clear()` (a `memset` of the hash buckets), `gmma.clear()`, the 16 barriers and `tmem.reset()` the rest; libc's `memset` of the register file (3 KB, also `rep stos`) another 4.7% of the program. A zeroing loop with AVX2 or SSE2 stores is not faster than `rep stos` on this core (microbenchmark, 2.4 KB: 31 ns AVX2, 36 ns `rep stosb`, 50 ns SSE2), so only writing less helps.

### Part 1: chunks from huge-page arenas, and a bounded pool (`src/core/memory_backing.cpp`)
* Heap chunks are carved from 64 MiB arenas (`mmap`, 2 MiB aligned, `madvise(MADV_HUGEPAGE)`), a fresh mapping reading as zero: no `new[]()`, no second pass writing zeros over memory the kernel has just zeroed, one fault per 2 MiB where the kernel has huge pages (and the same as before where it does not). A chunk that has no room in the pool goes back with `madvise(MADV_DONTNEED)` (zero on next touch, no memory until then; `memset` if the call fails) onto a cold list. Under AddressSanitizer chunks stay `new[]()`, so the red zones remain.
* The pool of cleared chunks keeps any chunk now (it was those with at most 32 KiB written, 64 of them), up to `Config::pool_max_bytes` (1 GiB; also never more than the RAM limit leaves beside the chunks in use). The clearing is outside the lock. The host-memory guard counts the pool as the host's: a look that finds the host short gives the pool back and looks again (one extra sample, only then), so a pool never makes an allocation fail or spill that would have succeeded without it. Memory in use (`ram_bytes()`), the zero contents, the error text and the sample counts are as before; peak RSS is unchanged (atomicCost 14.43 GB both, churn 531 MB both).

| program | runs | instr B/A | cycles B/A (p) | page faults A, B | task-clock B/A (p) | sys s A, B |
|---|---:|---:|---:|---|---:|---|
| HeCBench atomicCost | 15 | 1.0040 | 0.987 (0.0005) | 7.21 M, 5.41 M | 0.934 (0.004) | 7.56, 5.71 |
| churn 256 MiB x 8 (alloc, copy in, free) | 15 | 0.605 | **1.148** (<0.001) | 590 K, 66 K | **0.358** (<0.001) | 1.10, 0.16 |
| churn 3000 MiB x 4 | 15 | 0.571 | 0.905 (<0.001) | 3.84 M, 2.29 M | 0.756 (<0.001) | 6.19, 4.47 |
| tiny (1 MiB x 1) | 21 | 0.995 | 0.934 (<0.001) | 818, 567 | 0.901 (<0.001) | 0.004, 0.004 |
| churn 256 MiB x 8, A against A | 15 | 1.000 | 0.997 (0.53) | 590 K, 590 K | 1.001 (0.77) | 1.06, 1.07 |
| tiny, A against A | 21 | 1.0005 | 1.030 (0.26) | 819, 819 | 0.997 (0.52) | |

`cycles:u` counts user mode only. In the 256 MiB churn the work moved from the kernel (system time -86%, task-clock -64%, wall 2.5 s to 0.5 s) to the user side's first touches of huge pages, so user cycles rose 15%: read task-clock, not cycles, for that row. atomicCost is 6.6% less CPU time (wall -7% to -20% in single runs, noisy).

### Part 2: block set-up (`src/sass/exec.cpp`)
* `Warp::reset()` replaces `~Warp(); new Warp()` and the vector shuffle: it lists every member. The uniform registers are cleared up to `ur_extent` (64 words before sm_100, where UR63 is URZ and the 6-bit field cannot name more; 256 from sm_100) instead of 256; the 640 bytes of per-lane state only a few kernels use (`device_error`, `rpc`, `cga_target`) sit in `Warp::Rare`, reached only through `rare()`, which marks it, and cleared only if marked. `LaunchStats` is cleared in place (its counters are the words before `inst_by_opcode`). The 16 barriers are set again only if `Block::bar()` was called since; `mbar`, `gmma` and `tmem` only if not already empty (`TensorMemory::pristine()`). A `static_assert` pins `sizeof(Warp)`; a debug build checks every warp after `init_block` against a new `Warp`, member by member, which also proves the `ur_extent` bound (a uniform register written past it fails).
* Not done: the register file (3 KB, now the largest part of what is left, 5% of gaussian). Skipping rows needs the widest destination of every opcode (a tcgen05.ld writes 128 registers), which was not audited, and the file is sized from the kernel's declared count plus 8, so there is little to skip.

| program | runs | instr B/A | cycles B/A (p) | task-clock B/A (p) |
|---|---:|---:|---:|---:|
| rodinia gaussian (library with Part 2, pool experiment) | 15 | 0.9940 | **0.9626** (<0.001) | 1.010 (0.34) |
| rodinia gaussian (final library) | 15 | 0.9940 | **0.9673** (<0.001) | 0.969 (0.09) |
| rodinia gaussian, A against A | 15 | 1.0000 | 0.9984 (0.28) | 1.000 (0.97) |
| HeCBench blockAccess (19 ms run: start-up dominates) | 15 | 1.0030 | 0.9753 (0.014) | 0.977 (0.46) |
| blockAccess, A against A | 15 | 1.0000 | 0.9954 (0.15) | 0.991 (0.17) |
| polybench LU | 15 | 1.0075 | 0.984 (0.12) | 1.023 (0.46) |
| polybench LU, A against A | 15 | 1.0000 | 1.001 (0.90) | 1.003 (0.84) |

The share the survey measured (gaussian 12%, blockAccess 17%) was an upper bound: with the changes the function is 5.9% of gaussian, and the gain is 3.3 to 3.7% there. Instructions go **up** on the rows where blocks are many (LU +0.75%, atomicCost +0.4%): a `rep stos` is one instruction and the explicit stores are several, so cycles, not instructions, are the measure here. LU (6 to 10% claimed) is below noise (-1.6%, p=0.12) and is not claimed; FDTD-2D was not measured (one run takes over 2 minutes, a 15-run pair over an hour on this box). `blockAccess` is a 19 ms process: its 17% is not in the device phase.

### Tests (CMake-wired, framework-free; Debug build on server3, all pass)
* `test_memalloc`: `chunks_given_back_are_kept_cleared_and_counted` (20000 bytes dirty, every byte dirty, 64 KiB promised, the last byte), `the_pool_keeps_no_more_than_its_limit` (limit 3 chunks, 0, and the edge: room for two chunks less a byte keeps one, exactly two keeps two), `the_pool_stays_inside_the_ram_limit`, `chunks_across_arenas_are_distinct_independent_and_zero_and_come_back_zero_from_the_kernel` (1100 chunks, over an arena's 1024, written with a pattern each, handed back with no pool and taken again).
* `test_memory`: `host_guard_takes_the_pool_back_to_the_heap_when_the_host_is_short` (pool of 10 written-through chunks; plenty: one comes from the pool, none given back; short: pool emptied, two samples, refusal as before; one sample once the pool is empty; recovery: the chunks come out zero).
* `test_sass_block_reuse`: `uniform_registers_start_at_zero_in_the_next_block_and_the_next_launch` (UR62, the highest a program can write, and UR0, set by one kernel and read by the next, across blocks of 1, 2 and 4 warps and launches) and `a_blocks_counts_start_at_zero_in_the_next_block_and_the_next_launch`.

### Mutations (each rebuilt in Debug and run; "killed" = a test fails)
* Killed: no clearing of a chunk on release (6 tests in `test_memalloc`/`test_memory`); pool limit ignored; RAM-limit clause removed; `<` for `<=` at the pool's edge; no `MADV_DONTNEED` for a chunk past the pool; no pool give-back when the host is short; no second look after the give-back (the sample count); arena step of half a chunk; `ur` not cleared; `ur_extent` 32 instead of 64 (all by the new tests, `test_sass_block_reuse` for the last two).
* Run again on the tree merged with `origin/main` (round sp22-zero), one at a time, each rebuilt, with the tests that failed listed by `mut.py`; all 15 killed:

| mutant | killed by |
|---|---|
| `LaunchStats` not cleared in place | `a_blocks_counts_start_at_zero...` (also in a Release build: the counts are wrong) |
| `up[7]` not set / `p[7]` not set / `store_gate` not set in `Warp::reset` | every kernel test, by `expect_new_warp` (Debug) |
| `rare()` does not mark `rare_dirty` | `the_return_address_register_starts_at_zero...` (RPCMOV, so it is reachable on sm_90: the state a Release build also shows) |
| `ur_extent` 64 for every architecture; 128 from sm_100 | `uniform_registers_past_63_start_at_zero_from_sm_100` (UR200 and UR254, read back by the next kernel: visible in Release) |
| `Block::bar()` does not mark; the barriers never set again | `a_barrier_left_part_full...` (the next kernel's four warps deadlock on a barrier that kept one warp: Release too) |
| `mbar` never cleared | `an_mbarrier_a_block_used_is_gone...` (the phase bit of the word after the last arrival: Release too) |
| `gmma` never cleared | `a_warpgroup_mma_only_some_warps_issued...` (Debug only: the check in `init_block`) |
| `tmem` never reset | `tensor_memory_a_block_allocated_is_free...` (the granule the next block is given: Release too) |
| `pristine()` ignoring `live`, a collector's `desc`/`b_fields`, or `peer_dealloc` | `tensor_memory_is_pristine_only_when_every_member_is_what_reset_leaves` |

  Because a thread keeps one `Block` across the tests of a process, a mutant that leaves state behind also fails the tests that run after the one aimed at it; the table names the aimed one.

### New tests (round sp22-zero)
* `test_sass_block_reuse` runs hand-built SASS for sm_86, sm_90 and sm_100 now (`Env(sm)`; the parameters stay at the default base, 0x160, because the cubins carry no PARAM_CBANK, and from sm_90 a store prints as `STG.E desc[UR4][...]`). Words come from the disassembler corpora in `nvidia/tests/data/sass`, or from them with a register or immediate changed; every word is checked against `sass::to_text` before it runs.
* A barrier left a quarter full (`BAR.ARV 1, 128` by one warp) must not be there for four warps that `BAR.SYNC 1, 128` (sm_86, 90, 100); an mbarrier with a transaction due (`SYNCS.ARRIVE ... A0TR`) must not hold the phase back from the next block's last arrival (sm_90, 100); one warp of a warpgroup issuing `HGMMA` and ending leaves a snapshot that the next block must not find (sm_90; checked by the Debug comparison only); `UTCATOMSWS.FIND_AND_SET` without a free is given granule 0 again by every block (sm_100); `RPCMOV` (the `Rare` state) and UR200/UR254 (sm_100) read back as zero in the next kernel; `TensorMemory::pristine()` is false for each member `reset()` restores and true after it.
* In a Debug build `init_block` now also compares the barriers, mbarriers, warpgroup snapshots and Tensor Memory with a new block's (the warps were compared already).

### Re-measured on the merge with `origin/main` dad11eee (round sp22-zero)
Same host and method as above (Release GCC 13.3 LTO, `VGPU_THREADS=1`, cpu 12, alternating A B, three library copies in rotation, `perf stat`), A = `origin/main` dad11eee, B = the merge. Both trees built on a tmpfs (the box's root disk is a USB drive under heavy I/O from other sessions, which starved the compiler's temporary files until `TMPDIR` pointed at tmpfs). Load average 14 to 25 on 16 CPUs throughout, so cycles are noisier than the instruction counts; the programs are the same or rebuilt from the same source (`churn` is a 20-line `cudaMalloc`, `cudaMemcpy`, `cudaFree` loop, rebuilt). Standard output with the timing lines dropped was identical in every run.

| program | runs | instr B/A (p) | cycles B/A (p) | task-clock B/A (p) | page faults A, B |
|---|---:|---:|---:|---:|---|
| HeCBench atomicCost | 9 | 1.0040 (<0.001) | 0.978 (0.005) | 0.938 (0.11) | 7.21 M, 5.41 M |
| rodinia gaussian -s 512 | 15 | 0.9939 (<0.001) | 0.964 (0.008) | 0.957 (0.07) | 1488, 962 |
| gaussian, A against A | 15 | 1.0000 (0.28) | 1.006 (0.74) | 0.999 (0.59) | 1487, 1487 |
| churn 256 MiB x 8 | 15 | 0.605 (<0.001) | 1.150 (<0.001) | **0.351** (<0.001) | 590 K, 66 K |
| churn 256 MiB x 8, A against A | 15 | 1.0000 (0.82) | 0.977 (0.12) | 0.951 (1.0) | 590 K, 590 K |
| churn 3000 MiB x 4 | 15 | 0.571 (<0.001) | 0.920 (<0.001) | 0.775 (<0.001) | 3.84 M, 2.29 M |
| tiny (1 MiB x 1) | 21 | 0.9951 (<0.001) | 0.953 (0.006) | 0.875 (<0.001) | 819, 566 |
| tiny, A against A | 21 | 1.0000 (0.29) | 0.981 (0.62) | 1.014 (0.83) | 819, 820 |

The gains survived the merge and agree with the first measurement (atomicCost cycles 0.987 then and 0.978 now, gaussian 0.967 and 0.964, tiny 0.934 and 0.953, the 256 MiB churn's task-clock 0.358 and 0.351). atomicCost has 9 runs and no A against A (one run is 40 s and 14 GB of host memory on a shared box); its cycle gain is p=0.005, and its task-clock gain is not significant here.

### Not verified, not done
* ASan, TSan and the e2e tests are CI-only (server3 has no CUDA toolkit; the e2e tests need nvcc). THP behaviour was measured with `enabled=madvise`; on a host with `never` the arenas still avoid the second write and the fault count is unchanged.
* The `HGMMA` snapshot guard is observed only by the Debug comparison; no Release-visible difference was found for a one-warp issue. Cluster-barrier state (`cga_target`, `device_error`) shares the `Rare` marking tested through RPCMOV, but no kernel here reaches `UCGABAR` or the device-runtime error path.
* The pool's 1 GiB default is not configurable by environment (a `Config` field only); it helps loops that allocate, fill and free buffers up to that size, not programs that keep what they touch.

## Round sp21 (rodinia-profile): shared-memory bank counting and IMAD.HI (branch `sp21-rodinia`, base `origin/main` 179c886f)

The question: where do the Rodinia rows and the stencil PolyBench rows (hotspot, needle, pathfinder, gaussian, lud, backprop, 2DCONV, 3DCONV, JACOBI2D, FDTD-2D) spend their instructions, and which handlers have avoidable work? Measured on server3 (Zen 4), one thread, SASS, `nvidia/rtx3060`, `perf record -e instructions:u -c 1000000` and `-e cycles:u -c 400000`, corpora of `tools/bench/real`.

### What the profile said (base build, share of instructions retired)
* needle: `step_warp` 23%, **`sass_bank_passes_offsets` 17%**, `h_shared<false>` 10%, `sass_bank_passes_widened` 4.5%. lud: `h_shared<false>` 18%, **`sass_bank_passes_offsets` 13%** (and 60% of its cycles are the program's own `lud_verify`). hotspot: `step_warp` 18%, bank passes 7%, `h_shared<false>` 7%. backprop: `step_warp` 17%, **`h_imad_hi` 9.3%**, `load_batch` 7%. gaussian: `step_warp` 14%, `sectors_hash` 8%, `init_block` 14% of the cycles (38% of it two `rep stos`: the register file and the shared array of every block). pathfinder and lud are mostly the program's own host code and libc `random`. 2DCONV, 3DCONV, JACOBI2D: `step_warp` 14 to 26%, `load_run` 6 to 12%, `warp_rows_k` 5 to 9%, `run_allocation` 4%, `count_run` 3 to 7%.
* `sass_bank_passes_offsets` counted, per call, from the entry that answered last, with a 32-entry table. Counting its calls (a temporary counter, not kept): needle 3.1 M calls, **21% of them misses** (each takes `sass_bank_passes_widened`, about 400 instructions) although the kernel has only some hundred shapes; lud 15.5 M calls, 88% full warps, the entry that answered last answered only 58% (a loop that alternates two shapes asks for the other one every time, and then pays the hash and a second compare); hotspot 85% last-entry hits. The misses of needle were not a capacity matter (a table of 128 changed nothing): the hash samples eight lanes of the key (1, 3, 6, 10, 15, 21, 26, 31), and a partial warp has most of those lanes at zero in the key (a lane that does not run is taken at the first lane's word), so the forty masks of a wavefront's diagonals were all one slot and evicted each other.
* `h_imad_hi` made a 128-bit sum for each lane (about 25 instructions a lane, scalar), for the high word of a * b + c: the wide multiply-add kernel (`imad_wide`) already computes that sum modulo 2^64 with the carry out.

### What changed
1. **Bank passes**: (a) the hash takes the lane mask, so shapes that differ in the mask alone no longer share a slot; (b) 128 entries for the 32-bit table (`BankMemo::kEntries32`; the 64-bit table keeps 32); (c) `MOp::way` (unused by a shared access until now) is the instruction's own entry, asked first instead of the one that answered last (`sass_bank_passes_offsets(..., hint)`; only ever a place to look first, read and written relaxed, masked to the table); (d) the partial-mask rows come from a 16-row table instead of being built from the nibble with five vector operations a group.
2. **IMAD.HI** through `c.sk->imad_wide` (low word thrown away, `rhi` stored, the carry out as before): the 32-bit addend extended by the sign of the product's kind, the negated 64-bit addend made in rows as `h_imad_wide` does.

### Results (counters and outputs identical in every row: `VGPU_COUNTERS=1` stdout/stderr hashed with the timing lines dropped)
Zen 4 (server3), one thread, cpu 4, `perf stat -e instructions:u,cycles:u`, alternating A B, 9 runs (FDTD-2D 3), medians, Mann-Whitney p on cycles; A = origin/main 179c886f, B = this branch, each built here and frozen. Load 1 to 9 from other sessions. A/A (A against itself, 9 runs): cycles 0.999 to 1.020 (backprop 1.020 p 0.10, JACOBI2D 1.010, the others 1.000 +- 0.002), instructions exactly 1.0000.
| program | instructions B/A | cycles B/A | p |
|---|---:|---:|---:|
| rodinia needle | 0.9436 | 0.9287 | <0.001 |
| rodinia backprop | 0.9267 | 0.9065 | <0.001 |
| rodinia lud | 0.9889 | 0.9866 | <0.001 |
| rodinia hotspot | 0.9992 | 0.9919 | 0.45 |
| rodinia gaussian | 1.0000 | 1.0044 | 0.06 |
| rodinia pathfinder | 1.0010 | 1.0061 | 0.20 |
| polybench 2DCONV | 1.0000 | 1.0155 | 0.10 |
| polybench 3DCONV | 1.0000 | 1.0134 | 0.007 |
| polybench JACOBI2D | 1.0000 | 0.9932 | 0.35 |
| polybench FDTD-2D (480 G cycles a run, 3 runs) | 1.0000 | 1.0203 | 0.05 |
| polybench ATAX | 1.0000 | 0.9995 | 0.57 |
| polybench GEMM | 1.0000 | 0.9916 | 0.12 |
The stencil rows have no shared memory and no IMAD.HI: their instructions are exactly equal and the +1 to +2% of cycles on 2DCONV, 3DCONV and FDTD-2D is where the linker put the code (the hot units moved with the edits to `tier_ops.inc` / `exec_ops.inc`; a first build that also had the change below read gaussian +3.1% and FDTD-2D +1.5%, the same lottery with another roll). Not fixed, as in the rounds before (see sp4 to sp10). `lint_hot_unit_layout` passes. The bench.py canary could not be run: server3 has no nvcc for its program (the key of the cached program includes the nvcc path).

### Tried and not kept
* `MemoryManager::run_out` / `run_in` (the 128-byte runs of the stencil loads and stores) four words to a step instead of one: 6 instructions per 8 bytes become about 2.75; instructions -2.2% on 2DCONV, 3DCONV, JACOBI2D and -2.2% on FDTD-2D, cycles -0.3% to -2.8% (3DCONV 0.975, p 0.002), but gaussian +3.1% in the same build with equal instructions (layout). With the change taken out gaussian reads +0.4% (p 0.06). Reverted (commit kept in the history, then its revert): a win in instructions that the cycles do not show on three of four rows and that costs a layout roll.
* A table of 128 entries alone (without the mask in the hash): needle's misses stay at 21%.
* Left alone: `init_block`'s two `rep stos` (gaussian; the register file and shared array of every block must be zero, so it is the memory traffic), `sectors_hash` (gaussian), `step_warp` (flat, see sp18).

### Tests (CMake-wired, no GPU) and mutations
* `test_bank_memo`: `an_entry_hint_is_only_a_place_to_look_first` (hints: per instruction, another instruction's, any byte, none; four shapes with different answers, masks drawn from four; the table must not enter its back-off), `warps_that_differ_in_the_mask_alone_do_not_all_share_one_slot` (sixteen masks over lanes the hash does not sample), `every_nibble_of_a_partial_mask_selects_its_own_lanes_in_the_key` (two masks that differ in one group of four lanes, every pair of 16 patterns, 8 groups, six random word assignments each: random masks never differ in one group alone).
* `test_sass_alu_runs`: `imad_hi_gives_the_interpreters_registers_and_counters_signed_unsigned_and_negated` (IMAD.HI and IMAD.HI.U32 with a pair addend R0:R1 .. R8:R9 or RZ, negated or not, 12 seeds, 32/33/100 threads, tier against `VGPU_FASTPATH=0`; also under `VGPU_TIER=0`). The 32-bit addend form (`!(aux & 4)`) is not reachable from an encoding this assembler makes and is not covered by a test; it is the `h_imad_wide` extension rule written the same way.
* Mutants built and run on server3 (each killed): hash without the mask (hash-collision test); partial-mask rows 5, 6 and 15 wrong, row 8 zeroed (nibble tests; rows 5/6/15 survived the first two versions of the test, which is why the pairs test exists); hint hit answering with the wrong entry's passes; a hint miss that skips the key compare; IMAD.HI: negation ignored, low word stored, sign ignored, negated high word shifted by 33. Survived, equivalent in results: the hint ignored in favour of the last entry (a performance change only; checked by the instruction counts above).
* Also run, all pass: `test_tier`, `test_sass_mem_rows` (and `VGPU_SIMD=base`), `test_memfast`, `test_sass_dispatch`, `test_sass_block_reuse`, `test_memalloc`, `test_exec`, `test_memhot_sp8`, `test_gather_sp10`, `test_sass_generic_mem` (and `VGPU_SIMD=base`), `test_warp_stride_access`, `test_sass_access_count`, `test_memrmw`, `test_memcopy_direct`, `test_fastpath`, `test_memcpy_chk`, `test_jit_resid`, `test_sass_atomics`, `test_sass_ldgsts`, `test_sched`, `lint_hot_unit_layout`. Not run: sanitizer builds, the AMD guards, the canary harness (CI).

## Round sp21 (address rows) (branch `sp21-addr`, base `main` 85ec63a5)

The question: how much of the SASS tier's instructions is a full warp's address row (`warp_rows_k`) and the `IMAD.WIDE` that makes the address, in programs whose accesses are runs or one address, and how much of it can go without a change of any result or counter?

### What the profile said first (A = main 85ec63a5, server3, `perf record -e instructions:u -c 20000000`, one thread, SASS, `nvidia/rtx3060`; share of instructions retired)
| program | `warp_rows_k` | `imad_wide_avx2` | `h_global` (both) | `step_warp` |
|---|---:|---:|---:|---:|
| babelstream (`-s 1048576 -n 3`) | 6.7% | 11.2% | 10.5% | 17.6% |
| aligned-types | 5.3% | (`iadd3x_k` 16.5%) | 7.9% | 26.9% |
| polybench 2MM | 15.3% | | 9.7% | |
| polybench GEMM | 16.3% | | 11.0% | |
| polybench FDTD-2D | 9.2% | 8.3% | 4.1% | 22.0% |
The survey's 23% for `warp_rows_k` in babelstream (a cycles figure) was **not reproduced**: it is 6.7% of the instructions here, and the cycles samples of this run (a 240 s cut under load from other jobs: 67 samples) are too few to say more than "about 6%". The 2MM, GEMM and FDTD-2D shares are as estimated (9 to 16%). `imad_wide` is the larger of the two in babelstream (11%) and FDTD-2D (8%), and alignedTypes is not an address-row program (its top is `iadd3x_k`).

### What changed
* **A run or a one-address load is asked of the registers' rows alone** (`simd::warp_run_kernel`, `warp_run_k`, AVX2; null at `VGPU_SIMD=base|off`). For a full warp the 32 lanes' base pair is lane 0's plus `l * step` modulo 2^64 (the added offset is the same for every lane and cancels), found eight lanes to a step in 32-bit arithmetic without making the 64-bit addresses and without storing them: the low words against `lo0 + l * step`, the high words against `hi0 + carry`, where the carry is the low sum being below `lo0` (`l * step` is below 2^32, so it wraps at most once); with no high row the carry must be zero. About 45 instructions where `warp_rows_k` is about 115 and stores 32 addresses. `step` is the access size (a run) or 0 (every lane the same address).
* **`h_global`** asks it first for a full warp (`MOp::shape`: the shape of the last full warp of this instruction, run / one address / neither, so that an instruction that is neither, a gather, pays nothing for it; relaxed atomic, only ever a place to start). A run goes straight to `load_run`/`store_run` with lane 0's address. A load of one address goes to `load_broadcast` and is counted by `count_global_progression` with stride 0, as before. When the memory manager declines, or the address is in the constant window, the address kernel runs as if this had not been tried (the registers are not yet written), so a decline, a fault and its error text are what they were. The addresses made after a successful one-address load, for the rare case that the closed-form count declines, are the one address (the destination may have overwritten the base).
* **`IMAD.WIDE` finds its carry-out only when a predicate takes it** (`imad_wide_k` with a null `carry_out`; `Carry` and `Cin` are template parameters, so the two 64-bit unsigned compares per half and the carry-in add are not made for an address computation). The sum is the same.
* Not done: **fusing `IMAD.WIDE` into the address kernel**. The pair is a register write that later instructions may read, and the tier has no liveness for registers, so the fused kernel would have to write it anyway; the saving is the one load of the pair, not the multiply. Not tried.

### Results (server3, one thread, cpu 10, `perf stat -e instructions:u,cycles:u`, alternating A B A' (A' = A from another copy of the library dir, the A/A control), 15 runs; A and B each in 3 fresh copies of the library dir, runs cycle through them; medians; Mann-Whitney p on cycles; stdout and `VGPU_COUNTERS=1` text identical in every row (a separate pass over 15 programs, below))
A = main 85ec63a5 built here and frozen, B = the branch. Load of the box 3 to 25 from other sessions during the runs; the A/A rows are the control.
| program | runs | instructions B/A | cycles B/A | p | A/A instructions / cycles (p) |
|---|---:|---:|---:|---:|---|
| babelstream (-s 4194304 -n 4) | 15 | 0.921 | 0.947 | <0.001 | 1.000 / 1.003 (0.82) |
| polybench GEMM | 15 | 0.927 | 0.957 | <0.001 | 1.000 / 1.003 (0.44) |
| polybench 3MM | 15 | 0.922 | 0.949 | <0.001 | 1.000 / 1.002 (0.82) |
| polybench GEMVER | 15 | 0.920 | 0.951 | <0.001 | 1.000 / 1.002 (0.82) |
| polybench GESUMMV | 15 | 0.961 | 0.966 | 0.012 | 1.000 / 0.990 (0.10) |
| polybench BICG | 15 | 0.947 | 0.962 | <0.001 | 1.000 / 0.999 (0.69) |
| polybench MVT | 15 | 0.943 | 0.957 | <0.001 | 1.000 / 1.003 (0.47) |
| polybench ATAX | 15 | 0.943 | 0.953 | <0.001 | 1.000 / 1.001 (0.37) |
| polybench 2DCONV | 15 | 0.948 | 0.977 | <0.001 | 1.000 / 1.007 (0.82) |
| polybench 3DCONV | 15 | 0.958 | 0.969 | <0.001 | 1.000 / 0.998 (0.58) |
| polybench JACOBI1D | 15 | 0.944 | 0.940 | <0.001 | 1.000 / 0.994 (0.44) |
| polybench JACOBI2D | 15 | 0.946 | 0.956 | <0.001 | 1.000 / 1.002 (0.42) |
| polybench SYRK | 15 | 0.959 | 0.970 | 0.001 | 1.000 / 1.007 (0.72) |
| polybench SYR2K | 15 | 0.954 | 0.966 | <0.001 | 1.000 / 0.999 (0.66) |
| polybench FDTD-2D | 7 | 0.933 | 0.953 | 0.009 | not run (4 minutes a run) |
| polybench 2MM | 7 | 0.924 | 0.979 | 0.11 (not significant) | not run (4 minutes a run) |

The instruction ratios are the reliable column (they moved by under 0.001 from run to run; every A/A instruction ratio is 1.000). Cycles are 3 to 5 points less than instructions, as in sp14 and sp18 (the gathers wait on L1 misses). The A/A cycle ratios are 0.990 to 1.007, so a cycle gain under 2% is not claimed: 2DCONV's 0.977 (A/A 1.007) is the smallest one that is. **2MM and FDTD-2D have 7 runs each, not 15, and no A/A, because a run is 4 minutes; 2MM's cycle ratio (0.979, p 0.11) is not a significant gain**, its instructions are. **alignedTypes was not A/B'd** (a run is over 20 minutes; its profile above says `warp_rows_k` is 5.3% of its instructions and the change touches little else there, so at most a few percent is to be expected). Rodinia (hotspot, gaussian, needle) is not in the corpus on server3 and was not measured; a program with no full-warp run or broadcast and no `IMAD.WIDE` would see none of this.

Where the gain comes from (instructions only, one run each, A, C = B with the `IMAD.WIDE` change taken out, B): babelstream 0.944 / 0.921, 2DCONV 0.958 / 0.948, SYRK 0.963 / 0.959, GEMM 0.928 / 0.927, 3MM 0.923 / 0.922, ATAX 0.943 / 0.943. The run and one-address test is most of it everywhere; the carry-less `IMAD.WIDE` is worth 2.3 points in babelstream, 1 in 2DCONV, and nothing where `IMAD.WIDE` is rare.

The estimate in the survey (23% and 10.5% in babelstream) was too high; the measured upper bound on this change was the share of `warp_rows_k` plus `imad_wide` (18% of instructions in babelstream, 24% in FDTD-2D), and the result is 7.9% fewer instructions in babelstream and 6.7% in FDTD-2D.

### Same bits
Standard output (timing lines dropped) and the `VGPU_COUNTERS=1` text of every kernel (instructions, lanes, global/shared/local operations, bytes, sectors, requests, bank conflicts, class mix) are identical between A and B for babelstream (62 kernels), GEMM, 2MM, 3MM, GEMVER, GESUMMV, BICG, MVT, ATAX, 2DCONV, 3DCONV (254 kernels), JACOBI1D (20,000 kernels), JACOBI2D, SYRK, SYR2K, ADI (2,049 kernels) and FDTD-2D (1,500 kernels), at the sizes above.

### Tests (CMake; framework-free `vtest`)
* `test_sass_mem_rows` (and its `VGPU_SIMD=base` twin): `warp_run_kernel_is_the_definition`, 400,000 random rows against the lane definition (the pair is lane 0's plus `l * width` modulo 2^64) for widths 0, 1, 2, 4 and 8, rows with and without a high row, first lanes within a warp's reach of 2^32 (the carry at every lane and at none), near 2^64 (the wrap), a lane off by one bit of either word, by an element, by a missing or extra carry; at least 50,000 runs, 50,000 non-runs and 20,000 carried runs are required.
* `test_sass_generic_mem` (now also as `test_sass_generic_mem_simd_base`, which the sp14 note said existed and did not): `full_warp_runs_at_the_end_of_the_allocation` (runs ending at the end of the allocation and 1, 4, 8, 32 bytes before and past it, every access size, guards; tier against the interpreter, the error text identical where the access faults, and it requires both outcomes); `global_accesses_over_several_warps` and `global_accesses_whose_shape_changes_from_warp_to_warp` (blocks of 64, 96 and 128 threads: the shape hint set by one warp is wrong for the next, mode 6: the first warp's lanes at one address, the later warps on a stride; full warps and the half warps a guard leaves; bytes, registers and every counter against the interpreter); `imad_wide_carry_out_predicate_guards_a_store` (lanes whose 64-bit sum wraps store under the carry-out predicate; 20, 32, 64 and 96 threads; the sums and the stores against the rule written out and the interpreter).
* `test_simd`: `imad_wide` over the eight combinations of signed/unsigned, carry-in mask or none, carry-out asked or not (a null `carry_out` is never written through).
* Also run, all pass, at the default level and `VGPU_SIMD=base`: `test_sass_alu_runs`, `test_sass_dispatch`, `test_sass_atomics`, `test_sass_block_reuse`, `test_sass_ldgsts`, `test_memcopy_direct`, `test_fastpath`, `test_exec`, `test_sass_lea`, `test_memcount_forms`, `test_memfast`, `test_tier`, `test_gather_sp10`, `test_warp_stride_access`, `test_sass_access_count` (36 runs of 18 programs). Sources build with `-Werror` on server3 (GCC 13) and compile with g++-12 `-fsyntax-only`.

### Mutations (a one-line change on server3, the test named failed; reverted)
Killed: run kernel's step `8 * bytes` to `8 * bytes + 1`; carry not subtracted from the high word; the no-high-row carry check removed; lane 0's high word not used (all `test_sass_mem_rows`). `h_global`: the run's base without `urv + imm`; the kernel asked with a width of 4 always; the one-address load counted with a stride of 4; the one-address load not sign-extended; its 8-byte high words not written (all `test_sass_generic_mem`). IMAD.WIDE: the tier asking for no carry ever (first survived, no test had a carry-out predicate through the tier: `imad_wide_carry_out_predicate_guards_a_store` was added and kills it); carry-in ignored when no carry-out is asked; carry-out from the first compare only (both `test_simd`).
Survived, equivalent: the run taken in the constant window with the first window test removed (the next test in the same condition, `g0 >= kWinEnd || g0 + n - 1 < kConstWindow`, says the same); a declined run not making the address rows (the older scalar pass that builds them when `rows` is false then does, so the answer is the same and only slower); the constant-window test of the one-address load (the memory manager has no allocation there, so `load_broadcast` declines and the batch path faults as before). The shape hint is a performance hint only: no result depends on it, so no test can tell.

### Not done / unverified
* `e2e_sass_tier`, `e2e_sass_archs`, the AMD guards, ASan, UBSan and TSan builds (no nvcc on server3, no sanitizer build run): CI. The shape byte is read and written with relaxed `__atomic` builtins, as `way` is passed to relaxed atomics, for TSan.
* alignedTypes and Rodinia not A/B'd (above); 2MM and FDTD-2D with 7 runs; the box carried load 3 to 25 from other sessions, so the cycle columns are less sharp than the instruction columns; the cycles profile of the first table was too sparse to use.
* A strided gather (ATAX's `A[i][k]`) still goes through the full address pass, which needs its addresses; a progression of any stride other than 0 and the access size would need its address fill when the access falls back, and the gain there is the 8 stores of a kernel that is about 115 instructions.
* The PTX engine and the AMD engine are untouched.

