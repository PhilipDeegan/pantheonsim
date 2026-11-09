## Tier 1 together (branch `speedup`: tier on by default, PTX atomics and texel reads, AMD baseline)

Measured on server2 (Ryzen 9 9950X, 32 logical CPUs, load 1.4 to 2.8 during the runs, CPUs 24-27, one host thread, `nvidia/rtx3060`,
CUDA 13.0.88, Release), `bench.sh --ab` base (`8f9eb524`) against the merged tree, 9 runs (SASS) and 11 runs (PTX), alternating. Counters
are identical in every row.

SASS engine (the default), B/A of the median, all p < 0.001 unless noted:
int_alu 2.75x, f32_fma 2.30x, f64_fma 1.51x, f64_subnormal 1.45x, wmma 1.09x, mma_ptx 1.10x, mem 1.41x, shared 1.39x, atomics 1.28x,
texture 1.12x, local 1.92x, divergent 1.94x, warp 3.38x, launch 1.21x; api 0.99x (p 0.014, under 2%: same) and startup 1.01x (same).

PTX engine (VGPU_SASS=0 and PTX-only modules): atomics 1.38x, texture 1.06x, mem 1.01x (same), shared 0.99x (same), **local 0.93x
(slower, p < 0.001)**. The PTX `local` regression comes with the memory-lookup change (`rmw_scalar`, `load_fields`); the branch that made
it measured 3% on a loaded machine and attributed it to code layout, but on a quiet one it is 7%. It is kept because the SASS engine is the
default on every NVIDIA GPU and PTX atomics gain 38%; the PTX local-memory path is the next section's subject (the cause, and a rewrite of the PTX local array that more than makes it up).

Dropped after measurement: the two-entry code hint and the bank-pass rewrite (see "Not done, and negative results"). On the AMD side the
harness (tools/bench-amd) counts a Radeon card's instructions with SQ_WAVE32_INSTS + SQ_WAVE64_INSTS, the counters a simulated RX 6800
offers.

## Tier 2: what the tier still hands back to the interpreter (branch `speedup2-tiercov`)

**Measured first.** `VGPU_TIER_STATS=2` (it already prints, per mnemonic, the straight-line instructions the interpreter ran and the ones that
ended a run and were the interpreter's) over the bench workloads (`quick` size) and over the corpora: HeCBench's `reduced.txt` (34 programs), the
Rodinia apps of `run.sh`, and the CUDA Samples of `exes.txt` (85 programs gave statistics; the others fail without a card or are not CUDA programs),
every one with the base build `a12f8d84`, `VGPU_THREADS=1`, the RTX 3060 profile. The tier ran 97.3% of the corpora's 5.98 G warp instructions
(HeCBench 97.2%, Rodinia 99.1%, Samples 97.4%); 0.34% were straight-line instructions with no handler and 2.31% ended a run in the interpreter.
Bench workloads: 100% for `int_alu`, `f32_fma`, `f64_subnormal`, `local`, `divergent`, `warp`, `launch`, `wmma`, `mma_ptx`; 99.8% `f64_fma`
(`I2F.F64.U32`), 99.5% `mem` (`UIMAD.WIDE.U32`), 98.9% `shared` (`P2R`, a waiting `EXIT`), 91.0% `texture` (`TEX`, `TLD`), 81.3% `atomics`
(`ATOMG.CAS`, `RED`, `REDUX`, `VOTEU`, `S2UR`). So the chain-ending instructions named in the notes above (divergent branches 35% of `divergent`'s
runs, barriers 25% of `shared`'s) are already the tier's: they end a *run* and return to the scheduler, they are not handed to the interpreter.

What is left, by dynamic count over all three corpora (share of the 5.98 G; the tier cannot do much about a count this small):

| instruction (mnemonic.modifiers) | warp instructions | share | where |
|---|---|---|---|
| IMMA.16816.U8.U8 | 33.6 M | 0.56% | one CUDA Sample (immaTensorCoreGemm) |
| LDSM.16.M88.2 / .4 | 12.6 M each | 0.21% each | the same sample |
| BRA.CONV | 11.5 M | 0.19% | atomicAggregate (3.1 M), CUB-style code |
| FCHK | 10.6 M | 0.18% | black-scholes 8.2 M, bilateral 1.2 M, silu |
| CALL.REL.NOINC / RET.REL.NODEC | 10.4 M / 8.0 M | 0.17% / 0.13% | the IEEE-division slow path of the same three programs |
| F2F.F64.F32 | 7.3 M | 0.12% | |
| ATOMS.CAST.SPIN.64 | 6.0 M | 0.10% | aop 4.4 M, atomicPerf 1.6 M |
| MATCH.ANY / .ANY.U64 | 4.1 M / 3.1 M | 0.07% / 0.05% | atomicAggregate |
| REDUX.OR, P2R, VOTEU.ANY, S2UR, RED.E.ADD | 3.4 to 4.1 M each | 0.06 to 0.07% each | atomicAggregate, atomicIntrinsics, radixsort2 |
| everything else (ATOMS.POPC.INC, ATOMG, R2UR, REDUX.MAX, BRA.DIV, CALL.ABS, ATOMS.ADD, R2P, MUFU.RCP64H, WARPSYNC, ...) | | under 0.04% each | |

No texture, BRX, generic LD/ST, 32-byte access, Turing global form, FRND or MUFU.RSQ64H stands above 0.01% in these corpora (TEX/TLD are 9% of the
`texture` bench workload only).

**What was kept: handlers for MATCH, IMMA and LDSM.** Each was chosen from the table above and kept because its A/B, on a program that runs it,
shows more than 2% with p <= 0.05. Each is a `tier::Form` with a handler in tier_ops.inc (form table entry O: integers and data movement, no number
format), the same results as the interpreter's code, and a fallback to `exec_mem` / the interpreter for anything unusual.

* `Form::Match` (`MATCH.ANY` / `.ALL`, 32 and 64 bits). The interpreter's code compares every lane's value with every other's through nested
  lambdas (about a thousand comparisons for a warp); the handler reads the lanes' values once and works out the lane mask of one equality
  class at a time, giving it to every lane of the class (`.ANY`), or compares each value with the first lane's (`.ALL`).
* `Form::Imma` (dense `IMMA`, 8- and 4-bit elements of both operands, s32 accumulator, plain or `.SAT`). `exec_mma` builds `std::vector<double>`
  matrices on the heap for every call and sums in doubles and int64; the handler gathers the fragments into two int16 matrices (B kept
  column by column), sums the products in 32-bit wrapping integers (the same bits as the interpreter's int64 sum cut to 32 bits; under `.SAT`
  in 64 bits, clamped to the s32 range after every 128 bits of K, as the interpreter does) and writes the executing lanes. Sparse IMMA (`.SP`)
  and the non-integer MMAs stay the interpreter's.
* `Form::Ldsm` (`LDSM` of 16-bit elements: `.M88`, `.MT88`, x1, x2, x4). A whole warp whose rows are all inside the block's own shared memory
  is moved straight into the registers (the window test of `h_shared`: another block's window, a row that could end past the array, and a partial
  warp, for which the interpreter reports a missing row address, all go to `exec_mem`). The 8-bit, 4-bit and 6-bit packed forms stay the
  interpreter's. The counters (`shared_loads`, `shared_bytes_read`) are the interpreter's.

| workload (what runs the instruction) | A | B | B/A median | B/A best | p | load |
|---|---|---|---|---|---|---|
| HeCBench atomicAggregate `main 1`: same build, `VGPU_TIER_OFF=match` (A) / not (B), 15 runs | 7.30 s | 5.19 s | 1.41x | 1.39x | <0.001 | 14 to 27 |
| the same, base build `a12f8d84` (A) against this build (B), 15 runs | 9.66 s | 5.26 s | 1.84x | 1.46x | <0.001 | 27 |
| CUDA Sample immaTensorCoreGemm: same build, `VGPU_TIER_OFF=imma` (A) / not (B), 5 runs | 57.2 s | 20.6 s | 2.78x | 2.83x | 0.009 | 7 |
| immaTensorCoreGemm: `VGPU_TIER_OFF=ldsm` (A) / not (B), IMMA on in both, 5 runs | 20.44 s | 17.52 s | 1.17x | 1.17x | 0.009 | 10 |
| CUDA Sample cudaTensorCoreGemm (HMMA): `VGPU_TIER_OFF=ldsm` (A) / not (B), 5 runs | 39.28 s | 35.35 s | 1.11x | 1.12x | 0.009 | 14 |

The three forms are 66.0 M of the corpora's 5.98 G warp instructions (1.1 points): the tier's share goes from 97.3% to about 98.4%.

(User plus system CPU seconds of the process, alternating runs, CPUs 16-19, `VGPU_THREADS=1`; 5 runs is the least that can reach p 0.009.) The
same-build rows remove any layout effect of the changed binary. MATCH is 3.8% of atomicAggregate's warp instructions and 29% of its time;
IMMA is 62% of immaTensorCoreGemm's warp instructions and about two thirds of its time.

**Tried and dropped, by the rule (an A/B that does not show more than 2% with p <= 0.05 on a workload that runs the instruction):**

* *A "direct" dispatch for the instructions that end a run and have no handler.* `tier_run` called `exec_mem`, `exec_warp`, `exec_control`,
  `exec_tex` or `exec_int` itself for ATOMG/ATOM/ATOMS/RED, REDUX, VOTEU, TEX/TLD, CALL/RET/BRX/WARPSYNC, BRA.DIV/.CONV and R2P/S2UR/R2UR, with the
  counters and the advance of `execute()` and without its guard, `fast_alu` probe and tracing check. It passed `e2e_sass_tier` and `test_tier`.
  The same build with `VGPU_TIER_OFF=direct` against without, 11 alternating runs (CPUs 28-31, load 11 to 28): atomicAggregate 1.089x median
  (best 1.027x, p 0.18), aop 1.031x (best 1.036x, p 0.053), atomicPerf 1.019x (best 1.032x, p 0.58); `bench.sh --ab` base against the build with it,
  11 runs: `atomics` 1.008x (p 0.047), `texture` 1.005x (p 0.047), both "same". These instructions spend their time in the memory manager's locks,
  `texture_fetch` and the lane loops, not in `execute()`'s prologue. Removed; nothing to keep.
* *A handler for FCHK* (the IEEE division's check; the interpreter answers "slow path" for every lane, so it is one predicate write). Same
  build, `VGPU_TIER_OFF=fchk` (A) against not (B), 15 alternating runs, CPUs 28-31, load 8 to 29: black-scholes 22.88 s vs 22.60 s, 1.012x (p 0.017);
  bilateral 1.736 s vs 1.705 s, 1.018x (p 0.011). Under the 2% line. Removed. (`sass_tier.cu` now runs an IEEE division so the
  FCHK, CALL and RET paths are held to the interpreter whatever is done to them later.)
* *CALL/RET, BRA.CONV/.DIV, REDUX, VOTEU, shared and global atomics, RED, TEX/TLD as handlers of their own.* The "direct" experiment above is
  the upper bound of what a handler for them can save when it only resolves operands once (the interpreter reads a register operand inline,
  so there is little to resolve); no handler was written for them. MATCH was the one whose interpreter code does more work than its
  result needs.
* *Whole-suite A/B of the kept changes.* `bench.sh --ab` base `a12f8d84` against this build, all SASS workloads, `--size full`, 9 runs,
  CPUs 16-19, load 3: counters identical in all, every verdict "same" (medians 0.98x to 1.01x, `wmma` and `mma_ptx` 0.998x and 0.997x;
  `launch` 0.970x median with p 0.008 but 0.988x best, `startup` is 2 ms of wall: code-layout noise of the kind this file documents
  elsewhere, not claimed either way). An earlier run of the MATCH-only build at load 20 had `mem` 1.035x and `launch` 1.028x "faster", which
  no MATCH explains.

**Same results with the tier off and on.** `tools/verify/run_corpora.sh` with `A` = the base build, `B` = this build, `T0` = this build with
`VGPU_TIER=0`: 28 of HeCBench's 34 programs (the other six, atomicReduction, babelstream, base64e, bh, binomial-small and atomicCost, time out
at 120 s in the base build, so `--skip-timeouts A` left them out) and Rodinia's 12, each run three times, and the CUDA Samples that run the new forms or the instructions around them (immaTensorCoreGemm,
cudaTensorCoreGemm, bf16TensorCoreGemm, warpAggregatedAtomicsCG, reductionMultiBlockCG, mergeSort, cdpAdvancedQuicksort, shfl_scan,
binaryPartitionCG, simpleAtomicIntrinsics, with a 500 s timeout). Exit status, stdout, stderr, output files and per-launch counters are
identical between B and T0 and between B and A except: `rodinia/backprop`'s `output.dat` (three different hashes in A, B and T0: the program is
not deterministic), `samples/UnifiedMemoryStreams` (host threads print in their own order), `atomicCost-cuda` (a stack address in an error
message), `samples/immaTensorCoreGemm` (a `TOPS:` rate line, 0.01 against 0.00 because B is 3x faster), and a `black-scholes-cuda` run of B
that the shared machine's memory ended with out-of-memory, which ran identically (B = T0 = A) when repeated.

**How the kept handlers are held.** `sass_tier.cu` gained `k_direct` (MATCH.ANY/.ALL/.ANY.U64 in whole and in diverged warps, blocks that are not
whole warps (128, 48 and 33 threads), and, beside it, the instructions the interpreter keeps: global and shared atomics of every kind and width,
RED, REDUX of every operation, the warp-aggregated increment's VOTEU/BRA.CONV, indirect and direct calls, `__syncwarp`, an IEEE division; each
block uses bins of its own so that four host threads give the one-thread answer), `k_imma` (mma.sync s8/u8/s4/u4 in m8n8k16, m16n8k16, m16n8k32,
m16n8k64, plain and `.satfinite`, on registers of random bits) and `k_ldsm` (ldmatrix x1, x2, x4, plain and `.trans`, rows shared between lanes).
`e2e_sass_tier` (every program run with `VGPU_TIER=0` and `1`) holds every result and counter to the interpreter's, also under four threads and
without the chain. `test_tier` requires MATCH to be taken in every corpus (sm_80, 86, 89, 90, 120), checks the operand rows and register windows
of IMMA and LDSM over every instruction of the disassembler corpora, and lists `match:O ldsm:O imma:O` in the form table. Mutation checks (a
scratch copy of tier_ops.inc, `e2e_sass_tier` on sm_86, each one caught by `sass_tier`): MATCH.ALL's predicate inverted, MATCH's 64-bit high word
ignored, MATCH.ALL's result written to all lanes; IMMA's A and B signedness exchanged, IMMA.SAT's clamp removed, the clamp interval doubled,
the accumulator dropped under `.SAT`; LDSM's transposed column index changed, `shared_loads` counted 31 for 32 lanes, the two halves of a
transposed word exchanged.
## Local memory layout and the PTX `local` regression (branch `speedup2-localmem`)

Measured on server2 (Ryzen 9 9950X, shared with other jobs: load 6 to 35 during the runs, so every row is an alternating A/B of 31 runs
(`bench.sh --ab`, 15 for the HeCBench program) on CPUs 22-23 whose SMT siblings were idle (checked with `mpstat` before each batch), one host
thread, `nvidia/rtx3060`, CUDA 13.0.88, Release with LTO, GCC 13.3). Builds: `base` = `8f9eb524`, `head` = `speedup` at `a12f8d84`,
`new` = this branch. Counters are identical in every row (the `counters` column of `bench.sh --ab`, and `VGPU_COUNTERS=1` output of the
HeCBench program diffed on both engines).

### Where the PTX `local` regression came from: code placement

The 7% (3% on a loaded machine) that arrived with `26e5d708` (`rmw_scalar`, `load_fields`) is not in any instruction the `local` workload runs:

* `libcudart.so` built at `c8ace690` (before) and at `8f9eb524` are byte-identical (`cmp`); an A/B of the two nevertheless read 0.967x median
  (best 1.005x, p < 0.001) once and 0.999x the next time. The harness's p-value does not see drift that is shared by the runs of one
  batch: a single batch on this machine is good for about +-3%, and the 7% of the Tier 1 text was one such batch. Alternating A/B of `base`
  and `post` (`26e5d708`), 41 runs each: post/base 0.895x (a noisy batch), 0.966x, 0.988x, and with A and B swapped base/post 1.030x, 1.030x, 1.028x: `post` is about 3% slower.
* The hot functions of the PTX `local` run, `exec_ld`, `exec_st`, `run_warp_until_yield`, `fast_int_bin` and `check_local`, are
  instruction for instruction the same in the two builds: `objdump -d` with addresses and call targets masked, 1885, 1557, 2249, 114 and
  127 lines, 0 lines differ. Their sizes are equal. They sit 0x660 bytes later (32 bytes modulo the 64-byte cache line:
  `exec_ld` 0x27c1c0 -> 0x27c820), because the texel and atomic code before them in the one LTO image grew.
* Putting 64-byte alignment on those three functions (`__attribute__((aligned(64)))`) took `post` from 0.97x to 0.987x, 0.986x, 0.990x of
  the base on `local`. On the tree of this branch the same pins gave `local` 1.014x, 1.007x but `shared` 0.957x and `mem` 0.981x: a
  different placement is faster for one workload and slower for another, so the pins are not kept. `-falign-functions=64` on all of
  `interpreter.cpp` (and, in a separate build, on all of `sass/exec.cpp`: SASS `shared` 0.93x, `local` 0.965x) was no better.
* The same holds the other way round: a build that changes only `interpreter.cpp` (this branch without its SASS edits; the SASS source is
  `head`'s) moves the SASS engine's `shared` to 0.93x and 0.935x of `head`, and the final tree's SASS `local` / `shared` read 0.964x /
  0.967x of `head` with no SASS source change at all (below). Any edit to the interpreter moves the placement of everything after it.

So there is no code to fix in `rmw_scalar` or `load_fields`, and nothing that a source change can promise to keep: the PTX `local` loss is a
property of where the linker happened to put code. What this branch does instead is make the PTX local-memory path itself cheaper, which
moves the workload far past the old 3%.

### The PTX local array, interleaved by words (`include/vgpu/exec/local_layout.hpp`)

`Warp::local` of the PTX interpreter was `std::vector<std::vector<uint8_t>>`: a vector per lane, each grown to the current frame on
its own, and every access went `lane_local(w, lane)` (an emptiness test, a `local_top()`, a size test) then a pointer chase to the lane's
heap block. It is now one array, `local_array(w)` (one size test per warp access, not per lane), laid out as the hardware does: byte `o`
of lane `l` at `(o / 4) * 128 + l * 4 + o % 4`. A warp-wide 4-byte access at one offset is 128 contiguous bytes. Growing the array for a
callee's frame is appending zero words (rows already there stay where they are). Every user of `Warp::local` in the PTX interpreter
goes through `exec::local_load` / `local_store` (fast paths: `exec_ld`, `exec_st`) or `load_routed` / `store_routed` (the general path);
the checks (`check_local`, the frame bounds, the alloca stack, which has its own per-lane buffers) and every message are unchanged, so
faults, out-of-range texts with offsets and counters are the same. Nothing else reads the array: no debugger, checkpoint or inspection
code touches `Warp::local`.

PTX engine, B/A of the medians (> 1: new is faster), p from the harness's Mann-Whitney test:

| workload | new / base (8f9eb524) | p | new / head (a12f8d84) | p |
| --- | ---: | ---: | ---: | ---: |
| local | **1.094x** | <0.001 | **1.141x** (best 1.098x) | <0.001 |
| shared | 1.004x (same) | 0.12 | 1.049x (best 1.005x, same) | 0.004 |
| mem | 1.007x (same) | 0.19 | 0.983x (same) | 0.89 |
| atomics | 1.375x | <0.001 | 0.983x (same) | 0.12 |
| texture | 1.119x | <0.001 | 1.041x (same) | 0.16 |
| divergent | 0.992x (same) | 0.57 | 1.011x (same) | 0.25 |
| HeCBench `base64e` (reduced: 1,572,864 elements, 4 cycles) | | | **1.198x** (best 1.204x) | <0.001 |

PTX `local` is back above the base (1.094x, with the atomics gain of 1.375x intact), and `base64e` is 1.20x. Other batches of the same comparison read `local` 1.09x to 1.11x of the base (three batches) and 1.11x to 1.18x of head (six), and
`base64e` 1.198x to 1.217x (five).

Why it wins: not cache lines. The warp's 8 KB of touched local memory is L1-resident in either layout; what was removed is the per-lane
indirection and the per-lane `lane_local` bookkeeping.

### Not done: the same layout in the SASS executor

The SASS executor (`Warp::local` in `src/sass/exec.cpp`, the loop in `exec_mem`, the tier's `h_local`, `mem_read` / `mem_write`) was
converted in a scratch build too, with the same helper, and with a second step on top: when all 32 lanes access one offset (a spilled
register, a constant index) the access is a 128-byte `memcpy` between the array and the register file instead of 32 lane moves. It does
not win, so it is not in this branch:

| SASS, `head` vs scratch build | local | base64e |
| --- | ---: | ---: |
| interleaved array, lane loops only | 0.997x, 0.999x (same) | 0.991x (p 0.007, same) |
| plus whole-warp row copies | 0.942x (p < 0.001, slower) | 1.001x, 1.028x, 1.045x (p 0.26, 0.03, 0.05) |

`local` indexes by lane-dependent values, so no warp-wide row exists there and the branch and the larger handler (`h_local` 1.4 KB ->
2.5 to 2.9 KB once each width has a row-copy and a lane-loop body) cost 6%; `base64e` stores its 65-byte table with 65 one-byte `STL`s at constant
offsets, the case the row copy is for, and gains at most 4% (not significant on a quiet run). The doc's expectation that touching 32 cache lines
was the cost did not hold: the working set fits in L1. The SASS executor keeps one frame per lane.

### Tests

* `tests/unit/test_locallayout.cpp` (`LocalLayout32Lanes`, `LocalLayoutOtherWarpWidths`): 60,000 random stores and loads per seed, of 1 to 16
  bytes at every offset (also across words), in random lanes, against a plain array per lane (the old layout), the array grown four times
  mid-run, then every byte read back and "a warp-wide word is contiguous" checked; 32, 64 and 8 lanes. Nine mutants of the header (a
  wrong row stride, the lane term, the in-word offset of a store and a load of 1 byte, the high word of 8 bytes in store and load, a 2-byte
  access at offset 3, the chunk length of the byte-wise copy, a 4-byte access at offset 1) are all caught (one by an abort).
* `nvidia/tests/e2e/sass_vector_memory.cu` gains `k_alias`: local memory written at one width and read at others (bytes, halves, words
  and pairs at lane-dependent places over a byte-filled array, and only some lanes), a callee handed a pointer into its caller's frame, and a
  recursive function whose per-call arrays (indexed by the seed, so they stay in local memory) must survive the callee's frames; the host
  runs the same body as the reference, so the layout is not in the oracle. `run_sass_vector_memory.sh` now also runs the SASS executor and
  the PTX interpreter with `VGPU_FASTPATH=0`, which are the only runs that reach `Runner::mem_read` / `mem_write` and the PTX
  `load_routed` / `store_routed` for local memory. The test passes on `head` too (it does not know the layout). A recursion that
  grows the array under live data was missing from the tests before: the PTX interpreter logged 96 -> 144 -> 192 -> 240 -> 288 growths on it.
* Mutation check of the PTX paths (each a one-line change, rebuilt and run through `run_sass_vector_memory.sh`): the fast load (a wrong
  element offset), the fast store (the element offset dropped), the routed load (+1), the routed store (low byte cleared) and the growth
  (the array replaced by a zero one) are each caught, by the `ptx` run (fast paths) or the `ptx-general` run (routed paths).
  The mutants of the discarded SASS half (fourteen of them, rows and lane loops of `exec_mem` and the tier, `mem_read`, `mem_write`) were all
  caught by their engines too.
* Run: the new unit test, all 208 `test_*` tests, `e2e_sass_vector_memory`, `e2e_sass_fastpath` and the 61 tests whose names contain
  sass, tier, ptx, fastpath, exec, fault, lockstep, local, vector, debug, diag, race, trap, assert, stack, call, device_functions,
  memcount, memfast or window: all pass but `e2e_lockstep_blocks`, which times out on a loaded server2 on `head` and `post` as well
  (passed and failed in alternate runs of the same build). `bench.sh --check-counters` is skipped on server2 (its nvcc builds other
  device code than the recorded counters), so the counters were compared between builds instead: `bench.sh --ab` over every workload
  and both engines (`counters identical`), and the HeCBench program's `VGPU_COUNTERS=1` output.

### Reading these numbers

The SASS rows of the same final tree against `head` (SASS source unchanged; interpreter edits only): local 0.964x, shared 0.967x, mem
1.027x, atomics 0.992x, texture 0.988x, divergent 0.994x, int_alu 0.996x, `base64e` 1.019x. The first two are the placement effect
described above (the same sources measured 0.998x and 0.93x in two builds that differed only in the PTX file). A merge that adds code
elsewhere changes them again; judge SASS changes on the merged tree, with an A/A of two copies of the same build next to the A/B.
## Tensor cores: the sm_86 chunked sum, faster (branch `speedup2-tensor`)

The card-measured sum of an RTX 3060's tensor core (`include/vgpu/exec/tc_ampere.hpp`, `VGPU_TC_EXACT`, on by default for sm_86) cost
3.5-4.5x over the plain sequential f32 model. Measured on server2 (`bench.py --ab`, base `speedup` a12f8d84 against this branch, CPUs
pinned, one host thread, `nvidia/rtx3060`), `VGPU_TC_EXACT=1` against `=0` on the base build: SASS wmma 0.199 s against 0.058 s of CPU, mma_ptx
0.102 s against 0.032 s. A sampling profile of the SASS `wmma` row (SIGPROF, `-g` build) put 66% of its samples in the vector kernel
(`mma_avx2_chunked<8>`) and 17% in the `hmma_k` wrapper around it: the model's own kernel, not the dispatch around it, is where the time goes.
The kernel alone, m16n8k16 f16, on one pinned core of a loaded box (a micro-benchmark that calls it 100,000 times; loads vary run to run,
so only same-process comparisons are quoted): 668-690 ns before.

What changed, all with the same result bits (the first is the test's oracle, a frozen copy of the earlier header):

* **Chunk length as a template constant** (8 or 4 products), so the product loops unroll.
* **Frame terms with a sentinel**: an operand that is zero gets a term of `-2^28` at decode time, so the largest frame position is a
  plain `max` of `ta + tb` per product instead of `mullo`, compare and blend; an all-zero chunk needs no special case.
* **One variable shift per product**: A's significands are kept times 16, so a product is already its magnitude `<< 4`, and the aligned term is
  that shifted right by `top - T` (truncation toward zero), computed in registers that do not spill (the products are formed again in the second
  pass, which measured faster than keeping sixteen vectors live: 566 -> 524 ns).
* **The sum's bit length from one float conversion** of its magnitude with the low 7 bits cleared (24 significant bits at most, so exact),
  replacing two double conversions and three permutes; the new accumulator stays in its register, and the lanes the vector code does not finish
  (an infinity or a NaN in the chunk, a result outside the f32 normal range) are the only ones that go through memory.
* **Four output rows at a time, stage by stage**: the 16 rows of an m16n8 tile are independent chains of about 60 cycles each, and one at a
  time the out-of-order window held about one. Rows per pass, same kernel, ns per call: 1: 527-549, 2: 480-522, 3: 433, **4: 421-425**, 6: 450,
  8: 413 (4 kept: it is the smallest that reaches the plateau, and 16 is a multiple of it; `VGPU_TC86_ROWS`).
* **An f16 accumulator** (mma/wmma `.f16` C and D) had no vector form and ran the scalar one, 8,776 ns per m16n8k16; the sum rounded to nearest even to
  f16 is now vector too (`sum_to_f16_avx2`), 552 ns.
* **The PTX engine's `mma.sync`** for the dense f16, bf16 and tf32 forms (f32 or f16 accumulator) goes through the SIMD kernel the SASS HMMA already
  uses (`Interpreter::exec_mma_simd`): the fragments are read register by register as before, gathered, and D is written the same way, instead of
  building `std::vector<double>` tiles with a switch per element.

### Results

`bench.sh --ab`, 11 to 15 alternating runs, counters identical in every row (the harness compares them), B/A of the median of warp-instructions per
CPU second, p from the harness's Mann-Whitney test. Loads are server2's, shared with other jobs (a quiet pair of CPUs was used where there was one):

| workload | engine | A (base) | B (this branch) | B/A | p | load |
|---|---|---|---|---|---|---|
| wmma | sass | 2.012 | 2.852 | 1.42x | < 0.001 | 7.8-10 |
| wmma | ptx | 1.179 | 1.672 | 1.42x | < 0.001 | 7.8-10 |
| mma_ptx | sass | 3.833 | 5.348 | 1.40x | 0.004 | 7.8-10 |
| mma_ptx | ptx | 0.847 | 0.945 | 1.12x | < 0.001 | 7.8-10 |
| (the same, with the PTX `mma.sync` route) mma_ptx | ptx | 0.885 | 2.165 | 2.45x | < 0.001 | 9 |

The final tree (all four commits), SASS engine, 21 alternating runs on CPUs 12-15, load 26-28: wmma 1.932 -> 2.802 (1.45x, best 1.50x, p < 0.001), mma_ptx 3.788 -> 5.651
(1.49x, best 1.48x, p < 0.001), counters identical.

A second set at load 25-28 with every CPU busy (so every number is lower and the ratios are less sharp): wmma sass 1.15x (best of runs 1.50x,
p 0.008), wmma ptx 1.46x, mma_ptx sass 1.48x, mma_ptx ptx 2.35x, all p < 0.001, counters identical.

An f16-accumulator program (`wmma` 16x16x16 and `mma.sync.m16n8k16` with half C and D, 6,000 iterations, 4 warps; CPU time of the process, 11
alternating runs, load 25-28, output hash identical): SASS 0.344 s -> 0.051 s (6.8x, p < 0.001), PTX 0.462 s -> 0.117 s (3.9x, p < 0.001).

### Same-bits tests (wired into CMake)

* `tests/unit/test_tc_ampere_fast.cpp`: the current `tc86::mma`, `mma_avx2` and the f16-accumulator form against `tests/unit/tc86_v0.hpp`, a
  frozen copy of the earlier header, over 26 million elements (13 shapes including M not a multiple of the row group, K not a multiple of the chunk, the
  largest 32 x 32 x 16; f16, bf16 and tf32; f32 and f16 accumulators; eight operand modes: plain, zeros, subnormals, infinities and NaNs of both kinds,
  largest finite values and powers of two, raw 32-bit patterns, cancelling pairs across chunk borders, one huge term among tiny ones; accumulators at,
  above and below the products' size). Also `sum_to_f32_avx2` and `sum_to_f16_avx2` lane by lane against the scalar conversions on every
  power-of-two neighbourhood, f16 grid ties and random sums for every frame.
* `tests/unit/test_mma_fast.cpp`: PTX `mma.sync` of every dense form with `VGPU_FASTPATH=1` and `=0` on sm_86, sm_80 and sm_90 profiles, half the
  lanes of a warp predicated off, 2.9 million words compared.
* Mutation checks (the test must fail), run by editing a copy of the header or `interpreter.cpp`: 33 mutants of the kernel (sentinel too small, the
  accumulator's frame position off by one, the `>> 4` bias, a dropped sign, `g` off by one, the bit-length mask, the remainder rows, the NaN/infinity masks
  of rows, columns and accumulator, the out-of-range redo, truncation replaced by rounding, the tie rules and overflow bound of the f16 rounding, the
  sign of a zero) and 7 of the PTX route (no sm_86 sum, accumulator type, C and D register order, f16/bf16 kind, a wrong K, a bf16/tf32 kind swap). 36 of 40
  are killed. The four survivors are all equivalent mutants: a low-bit mask of 63 instead of 127 in the sum's bit length (|S| < 2^30.2, so the
  top 24 bits start at or above bit 6), a wider "in range" test in the f16 sum (the shift it guards is never past -10 for a finite sum), and the
  PTX route with K cut to 8 or the bf16 and tf32 kinds swapped, which only make it decline (the fragment arity no longer matches) so the
  general path answers.

### Tests run

`ctest -j4` on the whole tree (629 tests): 97% passed. The 17 failures are not this change: 11 `amd_*` (no ROCm on server2), two lints that call
`git ls-files` in the rsync'd copy, `e2e_lockstep_blocks` (timed-out waits under the box's load; the base build fails the same way, run side by
side), and three nvcc CUTLASS compiles (`e2e_cutlass_hopper/gemm/sm100`) that were `Killed` mid-compile while Postgres held 21 GB (re-run alone, hopper was killed again and the other two were not completed). They need a
quiet machine or CI. Passed, among others: `test_tc_ampere`, `test_tc_ampere_fast`, `test_mma_fast`, `test_simd`, `test_fastpath`, `test_exec3`,
`e2e_mma_forms`, `e2e_mma_layout`, `e2e_wmma_types`, `e2e_wmma_gemm`, `e2e_float_half_forms`, `e2e_simd_lane_forms`, `e2e_sass_tier`.

### Not done

* **IMMA (int8/int4) and FP8 forms** still take the general element-by-element path (one `std::vector<double>` tile pair per instruction, a switch per
  element): about 2.3 us per m16n8k32 s8 `mma.sync`, 4 to 5 times the f16 route. An integer SIMD kernel is straightforward (exact arithmetic) but the
  `.satfinite` clamp after every 128 bits of K and the s4/u4/b1 layouts make it a piece of its own; none of the benchmark rows uses it.
* **Fusing the f16->f32 conversion and layout transposes of `hmma_k` into the decode of the exact kernel** (the wrapper is about a fifth of an
  HMMA now): not tried.
* **Sparse `mma.sp`, m8n8k4, block-scaled and narrow-float forms** are unchanged; they do not use the chunked sum (sm_86 only) except through
  the general path.
## The AMD engine: atomics across threads, contiguous lanes, and the matrix multiply (branch `speedup2-amdmem`)

Four of the ranked wins of "The AMD engine" (3, 5, 6 and 7). Three kept, one dropped. Measured on server2 (Ryzen 9 9950X, 32 logical
CPUs shared with other people's builds: 1-minute load 11 to 18 during the final A/B, pinned to the idle CPUs 8-15), Release, GCC 13.3,
ROCm 7.2 `hipcc` programs, simulated MI300X and RX 6800 only (no AMD card touched). `tools/bench-amd` A/B, base `speedup` a12f8d84
against this branch, 11 runs, alternating; ratios are B/A of the median (the best-of ratio agrees within 3% in every row but the 8-thread atomics, whose base runs 11 to 13 s and is the noisiest), all
workloads self-check bit for bit against their host reference.

| workload | target | threads | B/A median | B/A best | p | verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| atomics | mi300x | 8 | **8.66x** | 8.00x | < 0.001 | faster (wall 12.7 s to 1.5 s) |
| atomics | rx6800 | 8 | **4.27x** | 4.13x | < 0.001 | faster |
| atomics | mi300x | 1 | 1.008 | 1.015 | 0.33 | same |
| atomics | rx6800 | 1 | 1.032 | 1.041 | 0.10 | same |
| mfma | mi300x | 1 | **2.31x** | 2.30x | < 0.001 | faster |
| mfma | mi300x | 8 | **2.31x** | 2.33x | < 0.001 | faster |
| mem | mi300x | 1 | **1.18x** | 1.16x | < 0.001 | faster |
| mem | mi300x | 8 | 1.17x | 1.14x | < 0.001 | faster |
| mem | rx6800 | 1 | 1.12x | 1.11x | < 0.001 | faster |
| mem | rx6800 | 8 | 1.11x | 1.10x | < 0.001 | faster |
| valu_int (control) | mi300x / rx6800 | 1, 8 | 1.005 to 1.017 | | 0.06 to 0.56 | same |
| launch | mi300x / rx6800 | 1, 8 | 0.985 to 1.012 | | 0.24 to 0.75 | same |

The `valu_int` and `launch` rows are controls, code this branch does not change; they bound the noise of the run (about 2%).
`rocprofv3` wrote no counter CSV on this host during the session (the `--check-counters` path), so the harness was run with the
documented full-size instruction counts and its "counters identical" column is only the same number twice; the evidence that counts
and bits do not move is the unit tests below (they compare `DispatchStats` and memory images between the new and the old path) and
`ctest` over the AMD and memory tests (119 of 119, including the four RDNA/CDNA 5 targets of the tests that have them).

### What changed

* **Atomics across threads** (`MemoryManager::atomic_rmw`, `src/core/memory.cpp`). The striped `memory_atomic_lock` was taken for every
  lane of every atomic; at 8 threads `atomics` was slower than at one. With work-groups on several threads an atomic on an ordinary word
  of device memory is now a compare-and-swap on the word itself (4 and 8 bytes; the operation is applied to the value the swap
  replaces, and the result is stored only where the locked version stored it: when it differs, or, for buffer atomics, always, since
  that made a never-written chunk real). A chunk that is not real yet is made so first and the swap runs on it. Everything else (a
  fault armed, a host mapping, a word on the end of an allocation, an address that is no allocation's, a misaligned or 1/2/16-byte
  access) returns false having done nothing, and the caller takes the lock as before; one address always takes one road, so the two
  never meet on a word. One thread keeps its plain load and store with no lock. The scalar `s_atomic_*`, the vector and flat atomics,
  the buffer atomics and the host side of the hostcall ready stack (`Hostcall::service`) all use it, so a host thread and a kernel
  still exclude one another.
* **Matrix multiply** (`matrix_multiply`, `amd/src/mfma_math.hpp`). The binary16 widening was a call into libgcc (`__extendhfdf2`: 35%
  of the `mfma` samples; the baseline x86-64 has no F16C); it is now integer work, exact for all 65536 values including NaN payloads.
  The per-instruction heap vectors are per-thread buffers; B is kept by column so each dot product reads two contiguous runs; a shape's
  lane and output offsets (a handful of divisions by the shape's sizes each) are worked out once per thread and shape; binary16
  sources and a float accumulator in registers are read without the generic operand lambda; and float results are summed four at a time
  (four independent chains, each in the order it always was, so the bits do not change). A thread-local looked up per instruction
  cost 6% (`__tls_get_addr`); the machine holds the pointer.
* **Contiguous lanes** (`Machine::memory_run`, `MemoryManager::load_quad_run / store_quad_run`). A global load or store whose active
  lanes take consecutive addresses, one, two or four words a lane, goes to the memory manager as one `load_run` / `store_run` call (or
  the quad calls where each lane moves sixteen aligned bytes, which stay one 16-byte access each). Refused whenever lane by lane could
  differ (a fault armed, a host mapping, a run not inside one allocation, misaligned, peers attached), and then the lanes go one at a
  time as before and meet any error where they would. A load whose first address the lane before it just read does not start a run
  (the one-read-per-address rule of `load_once`), and the last lane's words become that rule's memory.
* **Dropped: cheaper one-wave launch set-up (7).** A launch's set-up is a work-group's register file, `Group`/`Wave` vectors and the
  hand-off (`launch` is 350 to 450 klaunch/s). The one removable piece, the wave's 64 KB zero-initialised register file (256 rows), costs
  0.7 us a launch measured in isolation on this host (`std::vector<std::array<uint32_t, 64>>(256)` built and freed, under load), against
  2.2 to 2.9 us a launch. Sizing it by the kernel's declared VGPR count is not bit-safe (any instruction may index a register past
  the declared count, as `s_set_gpr_idx` and a kernel's undefined reads do, and the file reads zero today), tracking what a wave wrote
  would add a store to every register access, and clearing a recycled file costs as much as making a new one. Nothing was shipped and
  `launch` is unchanged in the A/B above. (Superseded: "The cost of starting work", at the end of this file, found that the file's
  construction was 85% of `set_up_group`, not a third of a launch, and ships a pooled file that clears only the rows the code names.)

### Lessons

* The first version of the run path sat inside `global_access`'s lane loop, and `atomics` at **one** thread (which runs none of the new
  code) got 2.2 to 2.4% slower (31 runs, p < 0.003, twice); removing the block gave 1.000. The cause was the size of that loop, not the
  work: the same logic as a separate `[[gnu::noinline]]` member brought `atomics` back to 1.008 / 1.032 (same) and made `mem` faster than
  inline (1.18x, from 1.13x). Keep a new fast path out of a hot loop's body.
* The first matrix-multiply change (widening and the buffers only) measured 0.97x, slower: the generic operand lambda, not the
  conversion, was the cost. The profile (`tools/bench-amd/profile`) is what pointed at the per-instruction thread-local and the
  divisions.
* A mutation that is an equivalent program proves nothing: transposing the matrix plan's C offsets changes no result (it is only a
  storage map, read back through the same table), and the first version of the order test could not see a reversed dot product
  because a float result rounded from a double hides the order unless products cancel exactly (the test now uses powers of two whose
  products of 2^-28, 2^1 and 2^30 cancel or are swallowed according to the order).

### Tests

All wired into CMake. Each was shown to fail when the change is wrong (mutation checks, run on server2 against scratch copies):

| test | what it holds | mutations that fail it |
| --- | --- | --- |
| `test_memrmw_lockfree` (`tests/unit`) | `atomic_rmw` against load_scalar / operation / store_scalar over 120 000 random accesses (aligned or not, over the edges, freed, host-mapped, faults armed or cells stuck, uniform, real and never-written chunks, `always_store` both ways): values, memory left, chunks made real, error text; declines without calling the operation; six threads, no lock, adding, xor-ing and taking maxima on words in an unwritten, a uniform and a real chunk, final values equal the serial result; `load_quad_run / store_quad_run` against the quads one at a time | CAS replaced by a plain store; `always_store` ignored; a quad run reading a uniform chunk as zero |
| `test_amd_gcn_atomics` (extended) | 20 000 work-groups of `returning` (add, sub, xor, exchange) and 512 of `wide` (float and 64-bit add) on 8, 3 and 1 threads, lock-free and locked (`amd::set_fast_paths`): the words add up | CAS replaced by a plain store |
| `test_amd_gcn_memruns` (new, kernels `amd/tests/data/memruns.c`, built for gfx942, 1030, 1100, 1201, 1250; run on all five) | seven copy kernels (1, 2 and 4 words, 32- and 64-bit offsets) over 18 address patterns (consecutive, reversed, strided, repeated, overlapping stores, misaligned, over chunk and allocation edges, off the end, wild, random) and 7 lane masks, 3 draws each, new path on and off against twin memories: memory images, resident bytes, error text, wave / instruction / VMEM / flat counts | stores in reverse lane order; the exec mask ignored when extending a run; a load run landing in the wrong lane |
| `test_amd_mfma_math` (new) | all 65 536 binary16 values widened to the compiler's double, bit for bit (NaN payload, zero sign); the contiguous dot products against the strided ones they replaced; four at once against one at a time; order of summation | subnormal bias off by one; four-wide sums out of order |
| `test_amd_gcn_asm` (extended, kernels `asm_mfma_order.s`) | `v_mfma_f32_4x4x4_16b_f16`, `16x16x16_f16` and `32x32x8_f16` with a non-zero accumulator on random halves and on powers of two whose sums depend on the order, against the arithmetic written out from the layouts | B halves swapped; four-wide sums out of order |

## Tier 2 together (branch `speedup`, all pieces merged)

Merged tree against the Tier 1 tree (both Release, server2, CPUs 24-27 for NVIDIA and 24-31 for AMD, one host thread unless the AMD row
says `all`, 9 alternating runs, load 5-12). Counters are identical in every NVIDIA row; the AMD rows rest on the same-bits digest test
because rocprofv3 counting was unavailable on that host.

NVIDIA, B/A of the median: `wmma` SASS 1.48x, PTX 1.51x; `mma_ptx` SASS 1.47x, PTX 2.45x; PTX `local` 1.16x; `mem` SASS 1.03x;
`shared` SASS and PTX 1.02x. Everything else is within 2% (the Tier 1 work already took the rest). The tier's new MATCH, IMMA and LDSM
handlers show up only on programs that use them: atomicAggregate 1.4x, immaTensorCoreGemm 2.8x (IMMA) and 1.17x (LDSM).

AMD, B/A of the median against Tier 1's tree, p < 0.001 (simulated MI300X / RX 6800): valu_int 2.1 / 2.5x, valu_f32 2.1 / 2.7x,
valu_f64 1.9 / 2.5x, mfma 2.4x, mem 1.8 / 2.2x, lds 2.9 / 2.8x, atomics 1.8 / 2.3x at one thread and 12x / 8x with all threads, divergent
2.4 / 3.0x, wave 1.7 / 2.0x, launch 1.2x. api and startup are unchanged.

`test_amd_gcn_same_bits` (15,566 instruction shapes against digests recorded from the unmodified base) is the guard on all of the AMD
changes: the first merge failed it on 65 MFMA shapes (a different NaN payload when both operands are NaNs) and the fix keeps the fast loops
and recomputes only NaN results in the base's operand order.

## The AMD matrix instructions, every form (branch `speedup4-amdmatrix`)

The first rewrite of `matrix_multiply` (`speedup2-amdmem`) covered binary16 inputs into float accumulators on CDNA. This one measures what
each other matrix form costs per instruction, and speeds the forms that cost most: integer and 8-bit-float MFMA, bfloat16, float, xf32,
double and sparse (`v_smfmac_*`) MFMA, the gfx950 f8f6f4 and block-scaled forms, RDNA3/4 `v_wmma_*`, and gfx1250's `v_wmma_*` /
`v_swmmac_*`. Results are bit for bit what they were: the guard is `test_amd_gcn_same_bits` (digests recorded from f81cc0c1: 16,304 shapes,
the base's 15,566 and 738 new ones that cover every matrix form), plus the new old-vs-new test below.

Measured on server2 (Ryzen 9 9950X, shared; load average 2 to 14 during the runs), Release, GCC 13.3, simulated MI300X, RX 7900 XTX and
RX 9070 XT, one host thread, CPUs 24-31 for the harness (under `flock /tmp/speedup-bench.lock`), 20-21 for the per-instruction probe.

### What each form cost (before)

`amd_matrix_bench` (new, `amd/tools/matrix-bench.cpp`) runs one instruction of the ISA corpora over and over on a wave whose registers hold
small finite numbers (`time_instruction`, flavour 4 of the same-bits probe), and prints nanoseconds. Median over the operand variants of
a mnemonic (`amd/tests/data/isa_corpus_matrix_*.txt` holds several per mnemonic: registers and accumulation registers, constants for C,
`cbsz`/`abid`, `neg_lo`, `clamp`), 3 alternating runs of base and new, 150 iterations each:

| instruction | before ns | after ns | |
| --- | ---: | ---: | ---: |
| `v_mfma_f32_16x16x16_f16` (gfx942) | 1776 | 582 | 3.0x |
| `v_mfma_f32_32x32x8_f16` | 4209 | 1178 | 3.6x |
| `v_mfma_f32_16x16x16_bf16` | 2148 | 489 | 4.4x |
| `v_mfma_f32_32x32x8_bf16` | 4747 | 1092 | 4.3x |
| `v_mfma_f32_16x16x4_f32` | 1136 | 348 | 3.3x |
| `v_mfma_f32_32x32x2_f32` | 3346 | 875 | 3.8x |
| `v_mfma_f32_16x16x8_xf32` | 1547 | 401 | 3.9x |
| `v_mfma_f64_16x16x4_f64` | 2490 | 654 | 3.8x |
| `v_mfma_i32_16x16x32_i8` | 5223 | 893 | 5.8x |
| `v_mfma_i32_32x32x16_i8` | 10660 | 1343 | 7.9x |
| `v_mfma_f32_16x16x32_fp8_fp8` | 7609 | 1133 | 6.7x |
| `v_mfma_f32_32x32x16_fp8_fp8` | 10530 | 1855 | 5.7x |
| `v_smfmac_f32_16x16x32_f16` | 3223 | 1057 | 3.0x |
| `v_smfmac_i32_16x16x64_i8` | 8454 | 1524 | 5.5x |
| `v_smfmac_f32_16x16x64_fp8_fp8` | 11218 | 1696 | 6.6x |
| `v_mfma_f32_16x16x128_f8f6f4` (gfx950) | 25121 | 3387 | 7.4x |
| `v_mfma_f32_32x32x64_f8f6f4` | 29822 | 4785 | 6.2x |
| `v_mfma_scale_f32_32x32x64_f8f6f4` | 31101 | 6116 | 5.1x |
| `v_wmma_f32_16x16x16_f16` (gfx1100) | 33172 | 1242 | 26.7x |
| `v_wmma_f16_16x16x16_f16` | 34596 | 2988 | 11.6x |
| `v_wmma_f32_16x16x16_bf16` | 10496 | 1434 | 7.3x |
| `v_wmma_i32_16x16x16_iu8` | 15105 | 1385 | 10.9x |
| `v_wmma_i32_16x16x16_iu4` | 16440 | 1556 | 10.6x |
| `v_wmma_f32_16x16x16_f16` (gfx1201) | 34113 | 1262 | 27.0x |
| `v_wmma_f32_16x16x16_fp8_fp8` | 38463 | 2192 | 17.6x |
| `v_wmma_i32_16x16x16_iu8` | 16198 | 2185 | 7.4x |
| `v_wmma_i32_16x16x32_iu4` | 31470 | 3422 | 9.2x |
| `v_wmma_f32_32x16x128_f4` (gfx1250, 20 iterations) | 783952 | 56200 | 13.9x |
| `v_wmma_f32_16x16x128_f8f6f4` (gfx1250) | 277056 | 25570 | 10.8x |
| `v_wmma_f32_16x16x32_f16` (gfx1250) | 81358 | 7956 | 10.2x |
| `v_wmma_i32_16x16x64_iu8` (gfx1250) | 63663 | 7272 | 8.8x |
| `v_swmmac_f32_16x16x64_f16` (gfx1250) | 89051 | 11309 | 7.9x |

The ranking before: RDNA `v_wmma_*` at 10 to 40 us an instruction was ten times anything else (it compared the type's name, as a
`std::string`, for each of the 2 x 16 x 16 x 16 element reads); the gfx1250 forms at 60 us to 0.8 ms were beyond that; then the 8-bit
float and integer MFMA and the sparse ones at 5 to 30 us (an element decoded by a chain of `type ==` tests, an `ldexp` for each 8-bit
float, a double for each integer); then the binary16, bfloat16 and float forms at 1 to 5 us. (The probe's numbers wobble by a few
percent with the load and the operand variants: variants whose destination overlaps a source turn the data into NaNs and infinities
after a few hundred repetitions, and the NaN redo costs the same before and after.)

### What changed

* **A tile at a time** (`matrix_multiply_fast`, `wmma_fast`; `matrix_multiply` and `wmma` keep the old element-at-a-time code as
  `matrix_multiply_reference` / `wmma_reference`, used for anything the fast path does not take: operands that are not registers,
  registers beyond the file, a sparse form whose A is not in registers). Each operand is decoded once, with its type fixed at compile time
  (`matrix_elem<T>`, `matrix_decode_a/b`): A into rows of K, B into rows of K by column, C into the tile; the 8-, 6- and 4-bit floats by
  a 256-entry table built from `f8_to_float` itself; then the whole tile is summed (`mfma_tile_f`, `mfma_tile_fma`, `mfma_tile_int` in
  `amd/src/mfma_math.hpp`) and written back through the shape's place table.
* **The sums**: every output is still a double sum of its products taken in order of k from the accumulator, rounded once to float, so the
  bits are those of `mfma_dot`; what changed is that 4 rows x 4 columns (SSE2, 2-wide) or 4 x 8 (AVX2, 4-wide, chosen at run time by
  `__builtin_cpu_supports`; no fused multiply-add, so each product and each sum is rounded as before) are worked out together,
  and A's scalars are broadcast against a row of B. A sum that ends as a NaN is redone in the fixed operand order of last round
  (`mfma_tile_ordered`), so NaN payloads are what they were. The double-precision forms use one hardware `fma` a term (`target("fma")`,
  chosen at run time) with the same redo. Integer forms are exact: int16 tiles and `pmaddwd` (SSE2) or `vpmaddwd` (AVX2), four rows by
  4 or 8 columns; the accumulator is added afterwards (wrap for MFMA, saturate for WMMA with CLAMP).
* **Widening**: binary16 operands go to doubles four lanes at a time (`half4_to_double`, in float: a rebias and a shift, the subnormals as
  mantissa x 2^-24, the NaN payload kept; exhaustively equal to the compiler's conversion for all 65,536 values), bfloat16, float and xf32
  by a shift or mask and `cvtps2pd`; integer tiles are filled from the register bytes without going through doubles. The scalar
  `half_bits_to_double` cost 1.8 ns a value, 512 of them 0.9 us of a 16x16x16 MFMA's 1.8 us.
* **RDNA3/4 `v_wmma_*`** (`wmma_fast`): the old code looked up the element type by comparing `std::string`s for every one of the
  2 x 16 x 16 x K element reads; now A (16 x K) and B (K x 16) are decoded once and summed as a 16 x 16 x K tile.
* **gfx1250 `v_wmma_*` / `v_swmmac_*`** (`wmma1250`): the matrices are decoded once (`adec`, `bdec`), each block of K (the whole of it, or
  one scale's 16 or 32) is summed as a tile, and the scale and the accumulator are applied to the block sums exactly as before.
* **Operand order of the NaN rule, made explicit.** Where two operands of an x86 add or multiply are NaNs the first one's payload wins and the
  compiler chooses which is first; three places depended on that choice and now spell it out, each matched to the recorded digests:
  the MFMA dot (B first of the multiply, the sum first of the add, last round); the WMMA dot (`WMMA_ORDER` 2: B first of the multiply,
  the product first of the add: found by trying all four against the digests: no digest wrong for 2, 19 to 79 wrong for each of the others); and
  the double-precision fma (`mfma_fma`: the first NaN of B, A, the sum, quieted). The last was found by a unit test: two copies of the
  same loop in one translation unit (`mfma_dot_fma` and the new tile code) gave different payloads for `fma(NaN, NaN, NaN)`, because
  GCC swaps the first two operands of `fma` freely. The reference code uses the same functions, so it cannot drift either.

### Results

`tools/bench-amd/bench.sh --build <base> --build <new> --threads 1 --size full --runs 9 --cpus 24-31 --no-counts`, base = f81cc0c1 (plus the
probe), new = this branch; ratio = B/A of the median (the best-of ratio agrees within 2% in every row); without the counting runs (rocprofv3
wrote no counter CSV here) the harness takes the count as 1 and the ratios are the ratios of the speeds; the work is identical (the
tests below are the evidence that bits and counts do not move). `mfma` is the existing workload; the other seven are new (`tools/bench-amd/kernels/bench_amd.hip`, `run_mfma_i8` ... `run_wmma_iu8`: each
self-checks `D = n x (A x B)` for small integers against the host, bit for bit, and fails the run otherwise); the RDNA targets are the new
`rx7900xtx` and `rx9070xt`, which the default `--target` (mi300x and rx6800) does not include.

| workload | target | what it runs | A cpu s | B cpu s | B/A median | B/A best | p | spread A / B |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mfma | mi300x | `v_mfma_f32_16x16x16_f16` (existing) | 0.910 | 0.358 | **2.54x** | 2.55x | < 0.001 | 1 / 3% |
| mfma_i8 | mi300x | `v_mfma_i32_16x16x32_i8` | 1.213 | 0.246 | **4.93x** | 4.99x | < 0.001 | 2 / 4% |
| mfma_fp8 | mi300x | `v_mfma_f32_16x16x32_fp8_fp8` | 2.050 | 0.314 | **6.53x** | 6.59x | < 0.001 | 46 / 66% |
| mfma_bf16 | mi300x | `v_mfma_f32_16x16x16_bf16` | 0.978 | 0.316 | **3.09x** | 3.09x | < 0.001 | 1 / 4% |
| mfma_f64 | mi300x | `v_mfma_f64_16x16x4_f64` | 1.312 | 0.395 | **3.32x** | 2.91x | < 0.001 | 43 / 115% |
| smfmac | mi300x | `v_smfmac_f32_16x16x32_f16` | 0.864 | 0.298 | **2.90x** | 2.93x | < 0.001 | 19 / 79% |
| wmma | rx7900xtx | `v_wmma_f32_16x16x16_f16` | 8.689 | 0.360 | **24.2x** | 24.3x | < 0.001 | 7 / 89% |
| wmma | rx9070xt | `v_wmma_f32_16x16x16_f16` | 9.581 | 0.345 | **27.7x** | 26.2x | < 0.001 | 10 / 80% |
| wmma_iu8 | rx7900xtx | `v_wmma_i32_16x16x16_iu8` | 6.124 | 0.711 | **8.61x** | 9.42x | < 0.001 | 32 / 36% |
| wmma_iu8 | rx9070xt | `v_wmma_i32_16x16x16_iu8` | 4.482 | 0.571 | **7.84x** | 6.92x | < 0.001 | 51 / 54% |

Final tree, 9 alternating runs, size full (64 waves, 4,000 to 8,000 instructions each), 1-minute load average before each workload 1.6 to
31.9 (median 4.0: other people's builds arrived during the f64 to iu8 rows, which is where the spreads are large; the ratios are
across alternating runs, so they carry the load of both builds alike). An earlier run of the same A/B on the tree before the
vector widening (lower load, 1.9 to 6.0, spreads 1 to 3% except bf16 and the RX 9070 XT's `wmma`) gave `mfma` 1.50x, `mfma_i8` 4.83x, `mfma_fp8` 6.56x,
`mfma_bf16` 2.22x, `mfma_f64` 3.17x, `smfmac` 2.03x, `wmma` 19.5x / 14.4x, `wmma_iu8` 8.94x / 6.78x, all p < 0.001, which shows what the widening
(`half4_to_double`, `cvtps2pd`) added: 1.5x to 2.5x on `mfma`, 2.2x to 3.1x on `mfma_bf16`, 2.0x to 2.9x on `smfmac`. The counters are identical in every row
(the harness compares the two builds' counts only with the counting runs, which are unavailable here; `test_amd_matrix_fast` compares
`DispatchStats` and every counter through the digest instead).

### Tests

All wired into CMake; each was shown to fail on a scratch copy when the code is wrong (mutation checks, server2):

| test | what it holds | mutations that fail it |
| --- | --- | --- |
| `test_amd_gcn_same_bits` (extended: four new MFMA/WMMA corpora and gfx1250's, `VGPU_SAME_BITS_REFERENCE=1` runs the reference path) | 738 more shapes (`amd/tests/data/isa_corpus_matrix_*.txt`, written by `amd/tools/matrix-asm.py`: every MFMA, SMFMAC and WMMA form of gfx942, gfx950, gfx1100, gfx1201 and gfx1250 in several operand variants), digests recorded from f81cc0c1 with the probe only; the 15,574 earlier digests are unchanged (the file is the base's plus the appended sections); passes on the tile path and on the reference path | k terms in reverse order (AVX2 kernel); NaN redo removed; wrong NaN operand order; fma NaN priority A first; integer pairs swapped; AVX2 kernel drops a row; fp8 table of the wrong format; sparse index set ignored; gfx12 WMMA half of K swapped; constant C ignored; A and B scales swapped; gfx1250 block offset ignored; gfx1250 sparse A not expanded; binary16 infinity exponent; subnormal scale; B step; xf32 mask; bfloat16 half; int8 sign; WMMA binary16 tile transposed |
| `test_amd_matrix_fast` (new) | every matrix instruction of eleven corpora (all the shapes above and the CDNA3, CDNA4, RDNA3 and RDNA4 ones from ROCm's libraries) on 8 seeds x 5 register flavours (any bits, small integers, floats with specials, half the lanes off, small finite numbers), tile path against the reference path through `set_matrix_fast_path`: registers, LDS, memory, counters, errors; and again with the registers of each instruction moved to random places that overlap one another (the destination over a source, the accumulator half over the destination, either bank) | the same list except the fma priority (both paths share `mfma_fma`; the digests and `test_amd_mfma_math` catch that one); and, in the second test only, the accumulator of an integer form read after the destination is written |
| `test_amd_mfma_math` (extended) | a tile against `mfma_dot` per output (random doubles of every magnitude, zeros, infinities, NaNs with random payloads) for 13 shapes; terms summed in order (1, 2^-60, -1); NaN payload for each of the four operand orders; fused tile against `mfma_dot_fma`; the priority of `mfma_fma`; integer tile against int64; the 4-wide kernels against the 2-wide ones (`mfma_avx2_switch`); all 65,536 binary16 values four at a time against `half_bits_to_double` | order, NaN redo, wrong order, integer pairs, AVX2 row, infinity exponent, subnormal scale |
| `test_amd_gcn_asm`, `test_amd_hipcc_mfma`, `amd_rocblas` (existing) | the layouts, order and rocBLAS GEMMs of the first rewrite | (unchanged) |

Run on server2 against this branch: `ctest -R ^test_amd` (143 of 143), `amd_rocblas`, `test_amd_hipcc_mfma_decode` (pass; `amd_hipsparselt` and
`amd_pytorch_xf32` skip there, for lack of the libraries).

### Found on the way

* **WMMA with CLAMP and a signed integer input is wrong in the reference code, and so on both paths** (left as it is, as the digests
  record it): `return is_signed ? static_cast<int8_t>(byte) : byte;` has the type `unsigned`, so a negative 8- or 4-bit element
  becomes 4294967295 - |v| + 1 as a double. Without CLAMP only the low 32 bits are kept and those are right; with CLAMP the int64 sum
  saturates the wrong way. `wmma_fast` hands exactly that case (CLAMP with a signed input) to the reference so the two keep agreeing.
  The fix is one cast; it changes results, so it is not in a speed-up branch.
* `mfma_dot_fma` gave a different NaN than an identical loop elsewhere (above); `part += a * bv` in `wmma_reference` and `wmma1250` had no
  fixed order; both are explicit now.

* Under `-fsanitize=address,undefined` (-O1, server2): `test_amd_mfma_math`, `test_amd_matrix_fast` and `test_amd_gcn_asm` have no
  AddressSanitizer report; UBSan reports one thing in the matrix code, the signed overflow of `isum += a * b` in `wmma_reference` (the
  signed-input bug above: 4294967234 * 4294967287); `test_amd_gcn_same_bits` at -O1 differs from the digests on 85 shapes, none of them
  matrix instructions (`v_pk_add_f32`, `v_fma_mixlo_f16`, `v_div_fixup_f16`, ...: float codegen that depends on the optimisation level, as the
  test's header says).

### Not done

* `v_wmma_f16_16x16x16_f16` / `v_wmma_bf16_*` (16-bit results) stay at 3 us: 256 `static_cast<_Float16>(double)` calls into libgcc,
  about 10 ns each, are most of it. A direct double to binary16 rounding would have to equal libgcc's on every double; not attempted.
* gfx12 `v_swmmac_*` (RDNA4's sparse WMMA) is not implemented by the interpreter at all (an error), so there is nothing to speed up.
* The per-instruction shape lookup (`matrix_shape(OpName(...))`, a hash and a string compare, about 50 ns) and the C gather / D scatter
  loops (about 2 ns an output) are now the larger part of the small forms (4x4x4, 16x16x4): not worked on.
* The fast path is not taken for a source that is not a register, for registers past the end of the file, or (WMMA) with CLAMP on a
  signed integer input; those run the reference code, unchanged.
* The harness has no gfx1250 target (hipcc has no gfx1250 programs here); those rows come from `amd_matrix_bench` only.
## AMD libraries and real workloads: packed halves, dot products, half conversions, and the code object reader (branch `speedup4-amdlibs`)

The AMD benchmarks above are synthetic. This piece profiled the interpreter on what actually runs on it -- rocBLAS (ROCm 7.2's, unmodified),
hipBLASLt, MIOpen, the pantheon GPU workloads built for gfx942, the HIP and library samples of `amd/tests` -- all on the simulated MI300X, and
kept the changes the profile and an A/B supported.

### What was measured, and how

* **Cost per instruction.** A scratch patch (not committed) wraps `Machine::step` in `rdtsc` and keeps, per mnemonic, the count, the CPU time
  inside `step` and the mean active lanes, written at exit (`VGPU_THREADS=1`). It adds about 15 to 20 ns per instruction (`s_waitcnt` reads
  19 to 22 ns), so the cheap forms are overstated and the expensive ones are not. Pinned to CPUs 24-31 (0-15 for the "after" columns; see the
  caveat in the last paragraph of this section).
* **Host functions.** `tools/bench-amd/profile` (SIGPROF), on a `-g` Release build.
* **Programs** (1.70 billion wave-instructions in all): `amd/tests/rocblas/blas` (saxpy, sdot, sgemm, dgemm, `gemm_ex` on halves, bfloat16s and
  bytes, complex); a scratch driver for larger rocBLAS calls (sgemm, dgemm, hgemm, `gemm_ex` f16, bf16 and i8, gemv, level 1, strided batched,
  trsm, syrk, cgemm, 512 on a side); a hipBLASLt matmul (f16, bf16, f32 at 1024); `MIOpenDriver` (forward, backward-data and backward-weights
  convolutions in fp32, fp16 and bf16, with the library's own search over its solvers; batch norm, pooling, softmax, activation, GEMM); the 46
  pantheon workloads (built with the flags of `amd/tools/run-pantheon-workloads.sh`, 5 s each); `amd/tests/hipcc` and `amd/tests/libraries`
  samples; hipSPARSELt's `spmm`. No PyTorch here, and no rocprofv3 counting on this host.

### Ranked: which instruction forms cost the time

Share of the wave-instructions ("count") and of the time inside `step` ("time"), and the cost of one execution (64 lanes). Forms that
another piece of this round owns are marked: (tier) the vector ALU and compare forms of `amdtier`, (mem) memory and LDS of `amdmemtext`,
(mfma) `amdmatrix`.

| suite | form | count | time | ns per execution |
| --- | --- | ---: | ---: | ---: |
| MIOpen fp16 convolution (its search tries the slow solvers) | `v_dot2_f32_f16` | 24% | 42% | 1390 |
| | `v_pk_fma_f16` | 3.0% | 13% | 3440 |
| | `buffer_load_short_d16` (mem) | 3.3% | 5.6% | 1346 |
| | `v_pk_mul_f16` / `v_pk_add_f16` | 1.1% / 1.0% | 4.2% / 3.3% | 2911 / 2518 |
| | `ds_read_b128` (mem) | 2.3% | 4.0% | 1388 |
| MIOpen fp32 convolution | `v_fmac_f32` (tier) | 40% | 42% | 299 |
| | `ds_read_b128` (mem) | 3.8% | 16% | 1227 |
| | the DPP forms (`v_fmac/add/mov_b32_dpp`) | 8.4% | 11% | 252 to 364 |
| | `ds_write_b32`, `buffer_load_dword` (mem) | 2.9%, 2.0% | 6.1%, 5.4% | 606, 763 |
| rocBLAS (blas + 14 larger calls) | `v_pk_fma_f16` (hgemm) | 3.9% | 28% | 3142 |
| | `v_mfma_f32_16x16x4_f32`, `v_mfma_f64_16x16x4_f64` (mfma) | 4.1%, 0.9% | 9.9%, 7.0% | 1070, 3625 |
| | `ds_read_b32/b64`, `buffer_load_*` (mem) | 7.2% | 10% | 552 to 1306 |
| hipBLASLt (f16, bf16, f32) | `v_mfma_*` (mfma) | 27% | 61% | 999 to 3536 |
| | `ds_read_*`, `ds_write_*`, `buffer_load_*` (mem) | 24% | 33% | 929 to 1230 |
| pantheon, 46 workloads | `v_lshl_add_u64` (tier) | 5.3% | 11.6% | 497 |
| | `v_pk_fma_f16` | 0.3% | 5.5% | 3824 |
| | `v_xor_b32`, `v_add_u32`, `v_mov_b32`, `v_fma_f32` (tier) | 6.4%, 2.6%, 3.9%, 2.3% | 5.2%, 4.3%, 4.0%, 4.6% | 186, 385, 239, 452 |
| | `global_load_dwordx4` / `dword` (mem) | 1.1% / 0.8% | 4.3% / 3.1% | 883 / 872 |
| | `v_mfma_f32_16x16x16_f16` (mfma) | 0.5% | 4.2% | 2114 |

By workload, the instruction that is most of the time: `compute_virus_agg` `v_pk_fma_f16` 69% (8144 ns) and `v_cmp_ngt_f16` 15% (7099 ns, tier's
compare path); `omni_virus` `v_pk_fma_f16` 77%; `tensor_virus` `v_pk_fma_f16` 70%; `transformer_train_step` `v_pk_fma_f32` 52% (tier);
`rope_stress` `v_sin_f32` 35% (1504 ns); `memory_retention_bake`, `pulse_virus`, `memory_thermal_asym` `v_fma_f32` 72 to 95% (tier); `fp64_virus`
`v_fma_f64` 88% (tier); `mma_virus`, `transformer_virus` `v_mfma_f32_16x16x16_f16` 93%, 84% (mfma); `atomic_virus` `global_atomic_add` 54% (mem).

The expensive non-memory forms, every one with at least 0.05% of all the time (cost of one execution, 64 lanes): `v_dot2_f32_f16` 1390 ns,
`v_pk_fma_f16` 3514, `v_pk_mul_f16` 2911, `v_pk_add_f16` 2518, `v_cvt_flr_i32_f32_e64` 2207, `v_fmaak_f32` 929 (tier), `v_add_f16_e64` 1182,
`v_subb_co_u32` 1419, `v_pk_fma_f32` 840 (tier), `v_cvt_f16_f32_sdwa` 1600, `v_sin_f32` 1504, `v_pk_mad_u16` 5110, `v_pk_max_f16` 2501,
`v_cmp_ngt_f16_e64` 7099 (tier), `v_fma_mix_f32` 797; against 190 to 300 for an ordinary ALU form. The packed-half and dot-product forms are
a few times the cost of anything else because every lane crosses libgcc's soft `_Float16` conversion (`__extendhfsf2`, `__truncsfhf2`: each a
call, three or four per lane per half) and the operation is chosen by string comparison in the lane loop.

**Host functions** (`blas`, 5 s, one thread, 2454 samples at 1 kHz): `Machine::step` 52% (`matrix_multiply` 12%, `vector_alu` 14%,
`lds_access` 7.5%, `buffer_access` 7%); **`hipModuleLoadData` 31%** -- 25 Tensile code objects of 0.5 to 20 MB inflated, of which
`load_code_object` 19.4% and `read_bundle` 8.9% (zstd 5%, the rest copies). A single 19.8 MB Tensile library
(`TensileLibrary_Type_DD_*_gfx942.co`, 689 kernels) took 5.5 ms to inflate and 25.5 ms to parse.

### What changed

All of it bit for bit; the new files are `amd/src/packed_math.hpp` and the tests below.

* **Packed 16-bit float arithmetic, a wave at a time** (`v_pk_fma_f16`, `v_pk_add_f16`, `v_pk_mul_f16`; `Machine::packed_f16_wave`).
  Halves are widened and narrowed with integer code that equals the compiler's conversion for all 65536 halves and all 2^32 floats (the
  libgcc call is gone); the multiply-add is the host's FMA unit's, and on a host with AVX2, F16C and FMA (checked at run time, and with
  MXCSR in its default state) eight lanes at a time. A lane with a NaN in it follows the old code's rule exactly: the first operand's payload
  for an add or a multiply (the order the recorded results show, held by inline assembly), libm's `fmaf` with the multiplicands in the
  recorded order for a fused multiply-add. Not taken for a MODE other than the default, gfx1250, a source that is not a vector register, an SGPR or a
  constant, a part-register destination: the old per-lane code runs those.
* **`v_dot2_f32_f16` and `v_dot2c_f32_f16`** the same way (products exact in double, summed in the order the old code summed them; AVX2 four
  doubles wide). A lane whose sum is a NaN is handed back to the old code.
* **`v_cvt_f16_f32`, `v_cvt_f32_f16`, and half comparisons** use the same integer conversions (`lane_half_f`); `write_half` tests the cheap
  `fp16_ovfl` before widening its result; the packed 16-bit integer instructions (`v_pk_mad_u16` and the other fourteen) resolve which
  arithmetic they are once per instruction instead of comparing the name in each half of each lane.
* **The code object reader** (`codeobject.cpp`): fields read with one bounds check, names as views of the file (no `std::string` per symbol),
  hash maps for the lookups (names are only looked up, never iterated), `std::vector` room for the kernels reserved up front (a `Kernel` is
  copied, not moved, when the vector grows), the MessagePack tree's arrays and maps reserved, `.text` copied with one `memcpy`.

### Results

Alternating A B, 9 runs after a warm-up, CPU seconds of the process (user + system), Mann-Whitney p, one host thread, CPUs 24-31 under
`flock /tmp/speedup-bench.lock`. A is `speedup` f81cc0c1 as built (Release, `-g`), B this branch. Outputs were compared (a hash of the
results printed by each program) and are equal in every row. The load average is that of the machine during the runs (others' builds).

| program | A median [best] | B median [best] | A/B median | best | p | load |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `v_pk_fma_f16`, finite halves (microbenchmark) | 1.133 s [1.077] | 0.070 s [0.060] | 16.2x | 18.0x | 0.0000 | 15 to 16 |
| `v_pk_fma_f16`, random bits (NaNs and infinities of every payload) | 1.250 s [1.171] | 0.094 s [0.077] | 13.3x | 15.3x | 0.0000 | 16 to 17 |
| `v_pk_mul_f16` (the loop is a third other instructions) | 1.363 s [1.331] | 0.967 s [0.881] | 1.41x | 1.51x | 0.0000 | 17 to 18 |
| `v_dot2c_f32_f16` | 1.287 s [1.244] | 0.300 s [0.249] | 4.29x | 5.00x | 0.0000 | 18 to 19 |
| rocBLAS `hgemm` 512 (whole process) | 4.433 s [3.894] | 0.848 s [0.809] | 5.22x | 4.82x | 0.0000 | 19 to 26 |
| `v_cvt_f16_f32` + `v_cvt_f32_f16` loop | 0.472 s [0.456] | 0.297 s [0.295] | 1.59x | 1.55x | 0.0000 | 9.5 to 10 |
| half comparisons (`v_cmp_{lt,ge,neq}_f16`) | 1.154 s [1.143] | 0.464 s [0.455] | 2.49x | 2.51x | 0.0000 | 7.6 to 8.9 |
| rocBLAS `blas` program (whole process, 25 Tensile objects) | 2.422 s [2.409] | 2.101 s [2.073] | 1.15x | 1.16x | 0.0000 | 12 to 19 |
| rocBLAS sgemm 256 (one library load) | 0.082 s [0.080] | 0.073 s [0.071] | 1.12x | 1.13x | 0.0000 | 11 |
| `load_code_object`, a 19.8 MB Tensile object (inflate + parse, 20 times) | 0.632 s | 0.310 s | 2.04x | 2.04x | 0.0000 | 7 |
| `load_code_object`, rocBLAS `Kernels.so` 0.83 MB (40 times) | 0.090 s | 0.053 s | 1.71x | 1.73x | 0.0000 | 7 |
| `v_sin_f32` + `v_cos_f32` (no change made: control) | 0.453 s | 0.450 s | 1.01x | 1.02x | 0.0040 | 10 |

Cost of one execution (64 lanes, the step-timing patch, the same program before and after): `v_pk_fma_f16` 3103 to 261 ns (rocBLAS hgemm:
3106 to 142), `v_pk_add_f16` 2464 to 248, `v_pk_mul_f16` 2304 to 148, `v_dot2c_f32_f16` 1130 to 149, `v_cvt_f16_f32` 789 to 293,
`v_cvt_f32_f16` 375 to 243, `v_cmp_*_f16_sdwa` 905, 894, 1277 to 512, 518, 428. The pantheon workloads run for a fixed 5 s, so what changes is how much
they do in it: wave-instructions executed in the 5 s, three alternating runs each, old / new: `omni_virus` 7.4 to 7.9 M / 24 to 34 M,
`tensor_virus` 11.1 to 12.0 M / 29 to 33 M, `compute_virus_agg` 5.0 to 5.3 M / 24 to 28 M; `llm_prefill` and `rope_stress` (no
change made to their forms) are the same within the run-to-run spread (15 to 22 M and 14 to 18 M both ways).

### Tests

* `test_amd_gcn_same_bits` (the 15,566 shapes of the ISA corpora against digests recorded from the unmodified base): passes unchanged.
* `test_amd_gcn_packed_same_bits` (new): 2,143 instructions of the families this changed -- `v_pk_{fma,add,mul}_f16`, `v_dot2*_f32_f16`,
  `v_cvt_f16_f32`/`v_cvt_f32_f16` with every output modifier, `v_cmp*_f16` with every condition and operand modifier, `v_sin/cos_f32`, the
  fourteen packed 16-bit integer instructions -- in every modifier combination (`amd/tools/packed-corpus.py`, assembled by llvm-mc, with raw
  VOP3P encodings for the op_sel/neg combinations the assembler refuses), run in the four flavours of the first test (random bits among them,
  so NaNs of every payload), against `amd/tests/data/packed_same_bits_digests.txt`, recorded from the unmodified base.
* `test_amd_packed_math` (new, 7 tests): the integer half conversions against the compiler's (all 65536 halves; all NaN floats; the
  rounding boundaries of every exponent; millions at random; `VGPU_PACKED_EXHAUSTIVE=1` runs all 2^32 floats -- it passed); the packed
  arithmetic against a copy of the old per-lane code on random waves with masks, constants, scalar sources, negation and clamp, with the host's
  FMA and libm's, and the AVX2 code against the scalar code lane for lane; the F16C conversion against the integer one for every NaN; the dot
  products (scalar and AVX2) against the old code.
* `test_amd_codeobject_same_bits` (new): the 241 code objects in the repository (the objects, hsaco files, and the bundles inside the hipcc
  executables) against digests of everything the reader returns, or of its error text, recorded from the unmodified base. 801 more -- every
  gfx942 object of rocBLAS, hipBLASLt and hipSPARSELt -- were compared old against new once: identical digests (`VGPU_CODEOBJECT_EXTRA=<file>`).
* Mutation checks, each of which makes the named test fail: the fused multiply-add's multiplicands in source order (both digest tests); the add's or the multiply's NaN order flipped (10 and 5 shapes of the first test); the AVX2 conversion rounding toward zero (both
  digest tests, two unit tests); the dot2c sum order (AVX2) or the dot2 sum order (scalar), neither of which the digests see, both the unit test;
  a rounding-boundary term dropped from the half narrowing (36 shapes and the unit test); global offsets 4 off in the reader (the code object
  test).
* ctest `amd|gcn|rocprof|hipcc|hip_` on server2 (a `-DVGPU_WERROR=ON` build of everything but a NVIDIA test that does not compile under
  -Werror on this base, `test_mma_fast`): all pass or skip but `amd_rocprofv3`, `amd_rocprofiler_abi` and `amd_tools` (host: no counter CSV,
  header version, user namespaces), as on the base; `amd_hipcc` and `amd_rocprof_first_calls_rx6700xt` failed once under `-j4` and passed alone.

### Not done, and negative results

* **Sine and cosine** (`v_sin_f32`, `v_cos_f32`, 35% of `rope_stress`): a double-precision sine with a three-part pi/2 reduction, a 25th-order
  Taylor series and a rounding-boundary check, which gave libm's rounded-to-float answer on every one of the 4.7 billion arguments
  checked (all floats under 2^14 turns, 91% answered; the rest fall back) -- and was **slower than glibc's `sin`** (18.7 ns against 11.9 per
  lane). Dropped, together with its header and test. glibc's table-driven sine cannot be beaten at libm's own answer.
* **A sweep of `static_cast<float>(lane_half(...))` over every site** was tried and reverted: it changed which NaN payload `v_fma_f16` and
  `v_div_fixup_f16` produce (the operands of the libm call reached it in another order), which the first digest test caught on 20 shapes. The
  conversion is used only where there is no arithmetic on the result (`v_cvt_f32_f16`, the comparisons, `write_half`'s test).
* **`v_pk_min_f16`, `v_pk_max_f16`** (2501 ns): their NaN and signed-zero results are `std::fmin`/`fmax`, whose operand order is the compiler's
  choice; left to the old code. **Dot product lanes with a NaN sum** go back to the old code too (the recorded digests hold only four dot shapes).
* **Integer dot products** (`v_dot4_i32_i8`, `v_dot4c`, `v_dot2_i32_i16`): measured at 195 to 310 ns, already an ALU form's cost.
* **DPP** (`*_dpp`, 252 to 364 ns, 4.5% of all the time): the shuffle adds 50 to 100 ns to an ordinary form; not worth a rewrite alone.
* **`v_div_scale/fmas/fixup_f32`** (800 to 1200 ns): 0.24% of all the time.
* **`carry_alu`** runs `op.find("_co_u32")` on every vector ALU form that reaches the generic path before its own ladder; a decode-time flag
  would remove it (touches `amd_gcn.hpp`/`gcn_decode.cpp`, which `amdtier` edits). Not measured.
* **`read_bundle`** still copies a bundle's code twice (the 4 MB window into the string, then `load_code_object` into the vector); 9% of the
  `blas` run. A zstd window written straight into place, or a `CodeObject` that keeps the bundle's bytes, would take most of it.
* Unexplained: `v_cvt_flr_i32_f32_e64` (2207 ns), `v_subb_co_u32` (1419) and `v_mul_f32` (3947 ns in the fp16 convolution) cost 5 to 15 times
  their siblings in MIOpen's OpenCL-compiled direct convolutions; the three have the same execution count, as though in one loop. Denormal
  operands would explain a float multiply but not an integer subtract.

Caveat on the "after" columns: they were measured on CPUs 0-15 (the benchmark cores were taken by other pieces' runs), so compare them with
the same-session A/B table above, not with the "before" columns' absolute values.

### Shared lines this touches (for the merge)

`amd/src/gcn_exec.cpp`: one `#include` after `mfma_math.hpp`; `lane_half_f` (new) after `lane_half`; one line of `write_half`; a block of new members
(`packed_halves`, `packed_plain`, `packed_f16_wave`, `dot2_f16_wave`) before `lane_float`; the packed integer block of `vector_alu`'s default
branch (rewritten in place); a prefix in each of `v_dot2_f32_f16`, `v_dot2c_f32_f16` and `v_pk_fma_f16`'s case; three lines each of
`v_cvt_f16_f32`, `v_cvt_f32_f16`; **one line of `Machine::compare`** (the half comparison, `amdtier`'s area). `CMakeLists.txt`: three test registrations after
`test_amd_mfma_math`.

## The AMD engine: memory instruction names read once, and the LDS accesses (branch `speedup4-amdmemtext`)

What was left of the text work on the memory instructions, and what the profile of the memory workloads showed after it.
Measured on server2 (Ryzen 9 9950X, 32 logical CPUs shared with other people's builds and benchmarks), Release, GCC 13.3, ROCm 7.2
`hipcc` programs, simulated MI300X and RX 6800 only (no AMD card touched), `tools/bench-amd` with a cache of its own, one host
thread unless the row says 8, pinned to CPUs 24-31, A and B alternating, 11 runs (9 for 8 threads), the benchmark lock held for each
run. Base: `speedup` f81cc0c1 built in its own directory. Every workload checks its result bit for bit against a host reference, and
the harness's "counters identical" column is `yes` in every row (the instruction counts of the two builds agree, quick size, through
rocprofv3 inside `vgpu shell`).

### What changed

* **Names parsed at decode, not at every execution** (`gcn::MemFacts`, `Inst::mem`, filled in by `gcn::finalize`). `narrow`,
  `half_access` and `parse_atomic` ran on the instruction's name text for every global, flat, scratch, buffer and LDS instruction
  executed, and `buffer_access` and the LDS ladder's default case ran them again for **every lane** (a `find("_d16")` per lane for
  every `ds_read_b64`, `ds_read2_b32`, `buffer_load_dword`...; a compare chain of up to sixteen strings per lane for a buffer atomic).
  `scalar_load` did a `find("prefetch")`, three prefix tests and, for `s_atomic_*`, a `std::string` and a chain of string compares.
  `mem_facts_of` works out, once, what the name says: the width and sign of a narrow access, the `_d16` forms, the atomic operation
  (global/flat, buffer and scalar atomics each have their own list of names), the prefix tests, the transposing loads, how
  `global_access` takes the instruction, how scalar memory does, and which LDS access it is. The parsers moved from `gcn_exec.cpp`
  to `gcn_decode.cpp` (one copy, `gcn::parse_narrow`, `parse_half`, `parse_atomic_body`, `transpose_bits_of`; the executor's
  `narrow`, `half_access` and `parse_atomic` call them) and an instruction nobody finalized (a test's hand-built one) gets the same
  facts from `gcn::mem_facts(in, scratch)`. The struct is 14 bytes (an `Inst` is 352): it is copied for VGPR indexing and gfx1250's VGPR MSBs.
* **The LDS accesses of real kernels without the ladder** (`Machine::lds_fast`): `ds_read_b32`, `ds_write_b32`, `ds_read_b64` /
  `_b96` / `_b128`, `ds_write_b64` / `_b128`, `ds_add_u32`, `ds_read2_b32`, `ds_write2_b32`, `ds_read_u8` / `i8` / `u16`,
  `ds_write_b8` / `b16`: one lane loop over the active lanes, the vector registers read and written directly, the same 32-bit
  address arithmetic and the same rules for an address past the work-group's LDS (a read gives zero, a write goes nowhere, a
  work-group given no LDS is caught, same text). Before it each lane went through a `switch` on the name's hash and then a text
  compare of the whole name (`OpName::operator==` was 8% of `ldsx`, a function call per lane), three more name tests, `lane_src`,
  `set_word` and `write_lane` calls, and, for the wide forms, the `half_access` scan. A fast path declines (the ladder runs) when
  an operand is not a plain vector register or a cross-lane instruction is running.
* **Global accesses read the address register directly** where it is a plain vector register (`lane_src` and `lane_src64` were
  separate calls per lane: 8-11% of the memory workloads), and so do the byte and short loads and stores; `memory_run`'s run
  detection reads the scalar base once instead of once per lane; a flat access finds its aperture (`flat_access`) with the scalar base,
  the immediate offset and the target worked out once. Same lanes, same order, same errors.
* **`Machine::step` no longer builds and tears down an `Inst` for every instruction.** `Inst remapped;` (gfx1250's VGPR MSBs) and
  `Inst indexed;` (the vector case, for VGPR indexing) were default-constructed on every step and every vector instruction: 2.3% +
  2.6% of the `smem` profile (`gcn::Inst::Inst`, `~Inst`), 4.8% of `divergent`. They are now a pointer made by a `noinline` copy where
  one is needed (`InstCopy`). The first version used `std::optional<Inst>`: with it `step` went from 6% to 21% of the `divergent` samples and `scalar_alu` and
  `more_alu` turned up as functions of their own (the inlined copy constructor and destructor made `step` larger, which is the likely
  reason the compiler stopped inlining them), and the whole engine measured 4 to 12% **slower** than the base (`valu_int`,
  `divergent`, `wave`, even `launch`; the controls caught it); out of line `step` is back to 6% and it measures faster. This part is `step`'s and the vector case's, not the memory instructions': the `amdtier` piece edits the same lines.

### Results (B/A of the median, base f81cc0c1 against this branch)

One thread, 11 runs, p < 0.001 unless noted. The rows from `ldsx` to `mem` and the `valu_int` control are one A/B (load average 3.5 to 6.6, median 5.2);
`atomics`, `valu_f32`, `divergent`, `wave` and `launch` are from another, base against the tree before the `flat_access` edit (load 1.9 to
16.6, median 4.3):

| workload | what it runs | mi300x | rx6800 |
| --- | --- | ---: | ---: |
| `ldsx` (new) | `ds_write_b128`, `ds_read_b96`, `ds_read2_b32`, `ds_add_u32` with barriers | **2.28x** | **1.99x** |
| `lds` | tree reduction, tile transpose | **2.14x** | **1.90x** |
| `buf` (new) | buffer loads and stores with offsets, some past the end | **1.33x** | **1.32x** |
| `flat` (new) | generic-pointer loads and stores that reach global memory | 1.15x | 1.14x |
| `smem` (new) | `s_load_dword`, `s_load_dwordx4` of uniform table entries | 1.11x | 1.19x |
| `narrow` (new) | byte and short global loads and stores | 1.10x | 1.09x |
| `mem` | 16-byte copy, scalar scale, gather | 1.10x | 1.12x |
| `atomics` | global atomic add, max, CAS loop | 1.13x | 1.18x |
| `valu_int`, `valu_f32` (controls: no memory instructions) | | 1.04x, 1.04x | 1.05x, 1.09x |
| `divergent` | | 1.09x (p 0.04) | 1.15x (p 0.02) |
| `wave` | | 1.01x (p 0.15, same) | 1.05x |
| `launch` | | 1.00x (same) | same (spread 60%) |

Eight threads (9 runs, load 2.1 to 6.0): `ldsx` 2.30x / 2.01x, `lds` 2.13x / 1.88x, `flat` 1.13x / 1.13x, `mem` 1.10x / 1.12x,
`atomics` 1.14x / 1.13x, `valu_int` 1.02x. The rows without a name for a change above are the `Inst` construction (all workloads) and
whatever the compiler made of the new layout; this table does not split them. What was split: the `scalar_load` facts alone, against
the same tree with the base's `scalar_load` text tests put back (15 runs, load 2 to 4): `smem` 1.034x on the MI300X and 1.050x on the
RX 6800 (p < 0.001; the `valu_int` control 0.99x and 1.00x, p 0.2 to 0.4).

### What the profile says is left

`MemoryManager::scalar_location` (15 to 19% of `buf`, `narrow`, `mem`: the chunk lookup of every lane's access), `load_scalar` and
`store_scalar`, then `lane_src` and `write_lane` called from `vector_alu`. The buffer path has no run detection (the global one does):
a lane-by-lane `load_words` for every lane of every `buffer_load_dword`. None of this is name text.

### Tests

Wired into CMake. Each was shown to fail when the change is wrong (mutation checks on scratch copies of the source, run on server2,
the build rebuilt each time; "caught" is a failing, crashing or silent test run).

* `test_amd_gcn_same_bits` (the existing guard, 15,566 shapes, digests from the unmodified base): passes unchanged. It runs memory
  instructions on random registers, so most of its addresses fault; it could not see most of the changes above.
* `test_amd_gcn_memory_same_bits` (new; `amd/tests/data/mem_same_bits_digests.txt`): every DS, flat, global, scratch, MUBUF, MTBUF and
  scalar memory shape of the ISA corpora (2,927 of them) run in twelve ways on a wave whose address registers are **aimed** at three
  allocations (`amd::digest_memory_instruction`, in `gcn_exec.cpp` next to `digest_instruction`): aligned, misaligned, on the end of an
  allocation, past it, wild, in the shared and the private aperture (gfx1250's too), runs of consecutive lanes, the same address in every
  lane, mixed per lane, half the lanes off; buffer resources valid, invalid, swizzled, scratch-resource, too small; LDS accesses past the
  work-group's LDS. Each of the first few shapes of a family (the atomic operations, the byte and short loads and stores, the LDS
  widths) is also run **under every name of its family**, so every atomic operation and width is reached. The digest holds registers,
  LDS, scratch, the three allocations, the instruction counts and the error text. The golden was recorded from a build of the **base**
  commit plus the probe function and nothing else (a scratch tree of f81cc0c1 with the function spliced in, run twice for
  determinism), never from the tree under test. About half of the (instruction, mode) pairs complete; the rest end in the error the
  base gives (invalid pointer, out of bounds, misaligned, past the LDS...) and so compare error texts. `s_memtime` and
  `s_memrealtime` (a clock) are left out.
* `test_amd_gcn_mem_facts` (new): `mem_facts_of` against the old per-execution tests, copied into the test as they were, for every
  instruction name of the corpora and for about 4,000 made-up names (every segment, form and suffix in combination); and the
  executor's fallback for an unfinalized instruction against the finalized one.
* `bench_amd_counters`: the new workloads have entries in `tools/bench-amd/expected/amd-hip-7.2.26015.json` (counts recorded with the
  base build; the existing entries are unchanged).

| mutation | caught by |
| --- | --- |
| `sbyte` loses its sign in `parse_narrow` | `test_amd_gcn_memory_same_bits`, `test_amd_gcn_mem_facts` |
| a `_x2` atomic's size not set to 8 in `parse_atomic_body` | both |
| narrow stores not told apart from loads in `global_access` | `memory_same_bits` |
| buffer `atomic_sub` adds | `memory_same_bits` |
| buffer `atomic_add_f32` with two NaNs takes the other operand's payload (see below) | `memory_same_bits` |
| scalar `s_atomic_umin` takes the maximum | `memory_same_bits` |
| scalar `s_load_i8` / `i16` with the sign inverted | `memory_same_bits` |
| `ds_read_i8` without its sign extension | `memory_same_bits` |
| `ds_write_b16` writes one byte | `memory_same_bits` |
| `ds_read2_b32` reads both words from the second offset | `memory_same_bits` |
| the `_d16` test of the LDS default case inverted | `memory_same_bits` (a crash) |
| atomics always four bytes | `memory_same_bits` |
| `scratch_store` sub-dword stores told apart from loads wrongly | `memory_same_bits` |
| narrow load written to the next register | `memory_same_bits` |
| run detection that over-reaches (`>=` for `==`) | `memory_same_bits` |
| flat access ignoring gfx1250's private aperture / the shared aperture | `memory_same_bits` |
| run detection and flat aperture detection that drop SCALE_OFFSET | **not caught**: the first only makes runs not be found (which is invisible, runs being exact), the second changes which aperture a lane is in only for addresses within a factor of 16 of an aperture's edge, which the probe's 32-bit offsets do not reach |

### Lessons

* **The two-NaN trap again.** The first run of the memory digests failed on one instruction, `buffer_atomic_add_f32` (gfx1250), where
  both the memory word and the register data are NaNs: x86 returns the first operand's payload, and which operand GCC makes the first
  is its choice. The base's build had the register's data first; the rewrite of the chain of `if`s into a `switch` put the other
  first. The rewrite now says it (`nan(b) && nan(v) ? v | quiet : b + v`), which is the base's bits.
* **A control that is not a control.** The first full A/B of the branch showed the memory workloads 10 to 120% faster and `valu_int`
  and `divergent` 6 to 12% slower; the controls were right. `std::optional<Inst>` in `step` was the cause (see above); out of line
  the same code is faster than the base for every workload. Always run the ALU workloads and `launch` next to the memory ones.
* **Tests that cannot reach the code.** The mutation of the `_d16` test in the LDS default case was first reported as not caught;
  the test had crashed, printing nothing, and the script looked for the word FAIL. And a mutation run left its object files newer
  than a restored source (rsync keeps the old modification time), so the next build used the mutant: touch after restoring.
* **A latent bug the probe found, left alone.** `scratch_at` checks `offset + bytes > scratch_per_lane`, which wraps for a negative
  `offset:` immediate with a small register offset (`scratch_store_b128 v9, ..., offset:-16` with v9 = 0): the check passes and
  the store lands before the lane's scratch, on the heap. The base does it, so does this branch (fixing it changes
  `test_amd_gcn_same_bits`'s recorded digests of those shapes, which may not be re-recorded); the probe skips scratch accesses with a
  negative offset. The fix is one line: `offset > per_lane || bytes > per_lane - offset`.

### Not done

* Scalar-load facts, `lds_fast`, the direct register reads and `memory_run`'s hoisted base were each kept only on the evidence of
  the table above; none was dropped. Not tried: a fast path for the rest of the ladder's LDS cases (`ds_read2st64`, the atomics with
  return, `ds_bpermute`), run detection for buffer loads, and moving `Inst plain;` out of `cross_lane_alu` (a default `Inst` per DPP
  instruction, the `wave` workload's remaining construction cost; it is the vector paths').


## The AMD pre-decoded tier (branch `speedup4-amdtier`, `VGPU_AMD_TIER`)

The AMD interpreter had been given opcode `switch`es and decode-time flags (the "AMD engine" sections above). What those left was the
frame every instruction still paid (`Inst remapped`, `FloatMode`, `debug::active()`, the mix counters, an `OpName` and a ladder per
helper) and, in every lane loop, the operand reads (`lane_src`, the `InvariantSources` table) and the EXEC test. This tier is the NVIDIA
tier's twin (`src/sass/tier*.cpp`): `gcn::finalize` calls `tier_classify` (`amd/src/gcn_tier.cpp`) when an instruction is decoded, which
resolves the hottest forms into a handler pointer and operand classes (`Inst::fast`, `amd/include/vgpu/amd_gcn_tier.hpp`): a VGPR row,
an SGPR, a constant, VCC, EXEC or the null register, each classified once; `Machine::step` calls the handler before it does anything else.
A handler works on whole register rows with kernels written once and compiled twice (`amd/src/gcn_tier_kernels.inc`: the baseline, and
under `#pragma GCC target("avx2,fma,popcnt")`, picked from `simd::level()` when the instruction is decoded). A scalar source becomes a
row filled once, EXEC is applied by one merge (a full EXEC writes straight into the destination), and a 64-bit operand is gathered from
its register pair. The part that needs the interpreter's `Machine` (global loads and stores) is `amd/src/gcn_tier_machine.inc`, included
into it: `global_access` with the work its name asked for done at decode, and the lane loop, `memory_run`, the load memo and every memory
call exactly as they were, in the same order.

**The contract is the tier's usual one**: a handler does the whole instruction, or nothing and says so, and the interpreter then runs it
as before. What it does not model it never claims (`tier_classify` leaves `fast.run` null): SDWA, DPP, VOPD, `op_sel` on a VOP3, 16-bit
halves, gfx1250, VGPR indexing and the MSBs (the wave's state is tested at run time), a `MODE` other than round-to-nearest with denormals
kept, tracing and the debugger. **NaNs.** Which of two NaN payloads wins depends on the order the compiler gave the operands, the trap the
MFMA work met, so a float handler that finds a NaN in an active lane of its result gives the instruction back before it has written
anything and the interpreter decides the bits; `v_min_f32` and `v_max_f32` also give back a pair of opposite zeros (which one wins is
libm's). A NaN *input* to a comparison, a convert or a min/max needs no such care, as none can pass a payload on. `neg`, `abs`, `omod` and
`clamp` are handled. Counters are those of the interpreter: `DispatchStats` is updated by the same instruction, before it can fail.

What has a handler (about 5,100 of the 15,500 decodable shapes of the ISA corpora):

* 32-bit vector integer and bit operations, VOP1/2/3 and their promoted forms with any scalar, constant, VCC or EXEC source: `v_mov_b32`,
  `v_not_b32`, `v_add_u32`, `v_sub_u32`, `v_subrev_u32` (not with `clamp`), `v_mul_lo_u32`, `v_mul_hi_u32/i32`, `v_mul_u32_u24`,
  `v_mul_i32_i24`, `v_mad_u32_u24`, `v_mad_i32_i24`, `v_and/or/xor_b32`, the three shifts, `v_min/max_u32/i32`, `v_bfi_b32`, `v_bfe_u32`,
  `v_alignbit/alignbyte_b32`, `v_add3_u32`, `v_or3_b32`, `v_xor3_b32`, `v_and_or_b32`, `v_xad_u32`, `v_lshl_or_b32`, `v_lshl_add_u32`,
  `v_add_lshl_u32`, `v_bcnt_u32_b32`, `v_mbcnt_lo/hi_u32_b32`, `v_med3_u32/i32`, `v_min3_u32`, `v_max3_u32`, `v_cvt_i32_f32`,
  `v_cvt_u32_f32`, `v_cvt_f32_u32/i32/ubyte0`, `v_cndmask_b32`, `v_readfirstlane_b32`, `v_readlane_b32`, `v_writelane_b32`; the carry
  family (`v_add_co_u32`, `v_addc_co_u32`, `v_sub_co_u32`, `v_subb_co_u32`, `v_subrev_co_u32`, `v_subbrev_co_u32`) and `v_mad_u64_u32`.
* binary32: `v_add/sub/subrev/mul_f32`, `v_fma_f32`, `v_fmac_f32`, `v_fmaak_f32`, `v_fmamk_f32`, `v_min/max_f32`, `v_rcp_f32`, and the packed
  `v_pk_add/mul/fma_f32`. binary64 and 64-bit integers: `v_add/mul/fma/fmac_f64`, `v_mov_b64`, `v_lshlrev/lshrrev_b64`, `v_ashrrev_i64`,
  `v_lshl_add_u64`, `v_cvt_f64_u32/i32`.
* `v_cmp` and `v_cmpx` of i32, u32, f32, i64, u64 and f64 (all sixteen float tests), into a scalar pair, VCC, a single SGPR (wave32) or EXEC.
* scalar ALU (`s_mov`, add and subtract with their carries, `s_mul_i32`, the logic family at 32 and 64 bits, shifts, `s_cselect`,
  `s_min/max`, `s_bcnt1`, `s_ff1`, `s_abs`, `s_bfe_u32`, `s_lshl1..4_add_u32`, `s_movk_i32`), the scalar compares, the `saveexec` family
  at 64 and 32 bits, the branches (`s_branch`, `s_cbranch_scc0/1`, `execz/nz`, `vccz/nz`), `s_waitcnt` (and its yield), `s_barrier` and the
  hints that do nothing.
* LDS (`ds_read/write_b32/b64/b128`, `ds_read2/write2[st64]_b32`, `ds_bpermute_b32`: a wave whose accesses do not all land in the LDS hands
  the instruction back) and global memory (`global_load/store_dword[x2,x3,x4]` and the byte and short loads and stores).

`VGPU_AMD_TIER_STATS=1` prints, at exit, the instructions the interpreter ran anyway. Over the bench workloads that is the per-wave
`s_load_*` and `s_endpgm` (about 1,000 of 570,000 on `valu_f32`), `global_atomic_*` (the `atomics` workload, which this piece leaves alone: the
earlier atomics round took its share) and `v_lshlrev_b32_sdwa`.

### Tests (wired into CMake)

* `test_amd_gcn_tier_same_bits` (`amd/tests/unit/test_gcn_tier_same_bits.cpp`): every instruction shape of the eight ISA corpora that has a
  handler (5,129 of the 15,500 that decode) is run through `amd::digest_instruction` with the tier off and on, in eleven flavours of
  pseudo-random state (numbered 0 to 4 and 10 to 15 after the merge with amdmatrix) and three seeds each (125,628 runs each way, about a minute): any bits; small integers; floats with the special values
  and NaNs; half the lanes off; two flavours of floats without a NaN (`10`, `11`; the tier's flavours were numbered 4 to 9 on their branch and moved to 10 to 15 on merging, because amdmatrix's probe flavour 4 took that number), so that the float handlers run instead of handing back;
  two (`12`, `13`) that put a second wave in the work-group and loads in flight, for the store gates and `s_waitcnt`; and two (`14`, `15`, for
  flat and global accesses only) that point the address registers into memory that exists, in consecutive lanes and scattered. The digest
  is everything the instruction changed or threw: all registers, masks, SCC, the program counter, LDS, memory, the wave's yield state,
  the counters and the text of an error. `VGPU_TIER_ONLY=<text>` restricts it to some instructions, `VGPU_TIER_SEEDS=n` runs more seeds.
* `test_amd_gcn_same_bits` (the guard; unchanged, digests recorded from the unmodified base) passes with the tier on, which is now the
  default, and with `VGPU_AMD_TIER=0`.
* Every `test_amd_gcn_*` unit test (25 binaries: real kernels, LDS, atomics, counters, memory runs, cross-lane, packed, doubles, spills)
  passes with the tier on and off alike (`VGPU_AMD_TIER=1` and `0`), and so do `test_amd_mfma_math` and `test_amd_hipcc_ops` (kernels built by `hipcc`).
* `amd_exec_bench` with `VGPU_BENCH_DUMP=1` prints all of a dispatch's counters (every `DispatchStats` field, the instruction mix) and a hash
  of the buffers it was given, filled with pseudo-random bytes. The nine bench kernels on the simulated MI300X (gfx942) and RX 6800
  (gfx1030): the counters and the hashes are identical with the tier on and off, and the two targets' buffers agree with each other.


### Why not `include/vgpu/simd`

The shared kernels (`simd.hpp`) were written for the NVIDIA executors and carry their semantics: a shift of 32 or more shifts every bit out
(AMD takes the count modulo 32), a NaN result is the canonical `0x7fffffff`, and `ffma` is fused only where PTX says so. The AMD interpreter's
own expressions are the specification here, so the tier has its own kernels (`gcn_tier_kernels.inc`, the same two-level scheme and the same
`simd::level()` choice, `fp-contract=off` for the file) and the tier-vs-interpreter test is what holds them to it. The first thing that test caught is an instance: the sign of the zero in `fmaxf(-0, +0)` (and `fminf`) differed between the interpreter's
compiled `v_max_f32` and the tier's, so the handlers give any pair of opposite zeros back to the interpreter.

### Mutation checks

Each of these was applied to a copy of the tree, built, and run against the new test (it must fail); the tree was then restored. All 19
fail (`FAIL`, with `VGPU_TIER_ONLY` set to the instruction concerned to keep each run short):

| # | mutation |
| --- | --- |
| M1 | `v_subrev_u32` with its operands swapped |
| M2 | a float handler keeps a NaN result |
| M3 | the integer handler writes lanes EXEC switched off |
| M4 | `s_cbranch_scc0` inverted |
| M5 | a comparison keeps the bits of inactive lanes |
| M6 | the LDS store gate dropped |
| M7 | a branch counted as a scalar instruction |
| M8 | `s_add_u32` loses its carry |
| M9 | `ds_bpermute_b32` reads a lane that is off |
| M10 | `v_cmpx` leaves EXEC alone |
| M11 | the global store gate dropped |
| M12 | `s_waitcnt` never yields |
| M13 | the reversed subtract in `v_subrev_co_u32` not reversed |
| M14 | the LDS range check loosened |
| M15 | `global_load_sbyte` loses its sign |
| M16 | the load memo dropped after a run |
| M17 | the last word of `global_load_dwordx2` lost |
| M18 | `global_store_short` stores a byte |
| M19 | a byte access sized as a dword when the runs are found |

M15 (run first, before flavours 14 and 15 existed) passed the real-kernel unit tests (`test_amd_gcn_memory`, `memruns`, `ops`, `lds`) with the
tier on, and the digest saw no global access with a valid address either (random registers fault): that is why flavours 14 and 15 (8 and 9 on the branch) exist, and
M15 to M19 were run with them.

### Results

Measured on server2 (Ryzen 9 9950X, load average before each workload 2.0 to 24.7, median 5.8, other people's builds and benchmarks
around, serialised with `flock /tmp/speedup-bench.lock`), Release, GCC 13.3, `hipcc` 7.2 programs, simulated MI300X (gfx942) and RX 6800
(gfx1030) only. `tools/bench-amd/bench.py` A/B, 9 runs alternating on CPUs 24-31, one host thread; base = `speedup` f81cc0c1 built
Release, B = this branch, `VGPU_AMD_TIER` unset (on). The host has no usable `rocprofv3` counter output, so the harness ran with
`--no-counts` and its rate column is 1e6 over the device phase's CPU seconds (the same count on both sides; the ratio is the speed-up);
what stands in for the counters is the bit-for-bit self-check of every workload and the `VGPU_BENCH_DUMP` comparison above. Ratios
are B/A of the median (the best-of ratio agrees within 5%), p is the Mann-Whitney p of the harness:

| workload | mi300x | rx6800 | p |
| --- | ---: | ---: | ---: |
| valu_int | **8.70x** | **8.09x** | < 0.001 |
| valu_f32 | **10.0x** | **7.93x** | < 0.001 |
| valu_f64 | **4.56x** | **5.27x** | < 0.001 |
| mem | **2.95x** | **3.22x** | < 0.001 |
| lds | **3.65x** | **3.17x** | < 0.001 |
| divergent | **4.39x** | **4.35x** | < 0.001 |
| wave | **11.2x** | **9.12x** | < 0.001 |
| atomics | **1.81x** | **1.96x** | < 0.001 (the global atomics are still the interpreter's) |
| launch | 1.12x | 1.10x | < 0.001 (kernels of a few instructions) |
| api | 1.01 | 1.02 | 1.0 / 0.26 (same) |
| startup | 1.00 | 0.99 | 0.93 / 0.22 (same) |

`api` and `startup` (not interpreter-bound) are the controls that bound the noise. With the tier **off** (`VGPU_AMD_TIER=0`, the same
build against the base): valu_int 0.966 / 0.965, mem 0.998 / 0.978, divergent 0.967 / 0.980, launch 0.986 / 1.001, startup 0.980 /
0.992: the opt-out is not free, about 2 to 3.5% on the instruction-bound workloads (the extra test in `step` and the larger `Inst`;
significant for valu_int on the RX 6800 and divergent, p <= 0.001), which does not matter as it exists to rule the tier out.

With all the host threads of the pinned CPUs (`VGPU_THREADS` unset, 7 runs, CPU time as above, p at the harness's floor of 0.001): valu_int
8.61 / 7.01x, valu_f32 9.69 / 6.70x, mem 3.00 / 3.25x, lds 3.65 / 3.14x, divergent 4.42 / 4.24x (MI300X / RX 6800); the work-group
threads share nothing the tier writes (the table of kernels is read-only, a handler touches its own wave, the bail counter is off unless a
test turns it on).

A first run of the same A/B, before the tier's per-instruction checks were hoisted into the `Machine` (the tier flag, tracing and the debugger
are now looked at once per dispatch instead of once per instruction; the other end of the same session's load differed), gave valu_int
8.28 / 6.82x, valu_f32 9.25 / 6.69x, valu_f64 4.51 / 5.10x, mem 2.96 / 3.12x, lds 4.08 / 3.03x, divergent 4.25 / 4.03x, wave 10.1 / 8.65x,
atomics 1.78 / 1.89x: within 5% of the table above except the RX 6800's valu rows (6.8 and 6.7 against 8.1 and 7.9). The second run is the
one reported; no claim is made about what the hoisting alone is worth.

### Not done

* `s_load_*` and `s_endpgm` (a few per wave: about 1,000 of 570,000 instructions of `valu_f32`), `global_atomic_*` (the `atomics`
  workload is 1.8 / 2.0x faster, from the instructions around the atomics; the atomics themselves are still the interpreter's), flat, scratch and buffer accesses, the narrow and 16-bit forms (SDWA, DPP, `op_sel`, halves, `v_cvt_f16_*`, `v_pk_*_f16`),
  the transcendentals (`v_exp_f32`, `v_log_f32`, `v_sqrt_f32`, `v_floor_f32` and the like), the division
  helpers (`v_div_scale`, `v_div_fmas`), MFMA (its own piece), gfx1250 (its own interceptions in `cdna5_alu`).
* A NaN-aware float path: a NaN result hands the instruction back, which costs the interpreter's time on a program that computes NaNs
  all the time (`VGPU_AMD_TIER_STATS=1` shows the count as "handed back").
* Profiling the tier itself (`perf` is unusable here): what is left of the per-instruction cost is the fetch from the decode cache, the
  `Regs` frame and the indirect call.


