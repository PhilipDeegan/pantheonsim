# Round 15 (sp33): tier 2 (the JIT) with control flow in generated code

Branch `sp33-jit` (from `origin/main` 7d8d98a4), 2026-11-07. Design: [06-tier2-jit.md](06-tier2-jit.md); what this is measured against: the study of 2026-11-06
(`~/.cache/pantheonsim-briefs/jit-study-2026-11-06.md`, called **the study** below) and today's JIT (rounds 11 to 14, called **base** below). The copy-and-patch question
is its own document: [copy-and-patch.md](copy-and-patch.md). Raw data: `data/round15/`.

## Verdict

**The goal is not met: geometric mean 1.099 over the 22 programs (goal 1.15); no program is more than 2 % slower (worst 0.982).** Warm cache, `VGPU_THREADS=1`, medians of 5 interleaved runs, server3; the study measured today's JIT on the same
programs at 1.034 (server2). The corpus total goes from 664 s to 628 s (1.057x): the long programs gain least. An A/A control (tier against tier, same binary, same protocol) over 15 of the programs gives 0.987 to 1.011 (geometric mean
0.9998), so a ratio outside 0.99 to 1.01 is a result and a ratio inside is not.

* **What it did.** Of the 22 programs, 18 now run regions; their geometric mean is 1.122 (simpleHyperQ 1.70, reduction 1.25, LU 1.18, FDTD-2D 1.17, transpose 1.17, SYR2K 1.14, SYRK 1.13, histogram 1.11, mergeSort 1.09, interval 1.09,
  alignedTypes 1.09, segmentationTreeThrust 1.09, atomicAggregate 1.09, sortingNetworks 1.06, gaussian 1.04). The generated code computes 55.0 % of all 18.1 G warp instructions of the 22 programs (the study, same programs: 27.9 %; 49.7 % was the ceiling of its opcode set over 112 programs),
  and 71.8 % of them run inside regions, escapes included (the study: 34.2 %). SYRK, the study's 12 % loss, is now +12.5 %: its 33.6 M region calls are 0.10 M.
* **Where it does not move.** Four programs have no region at all (convolutionSeparable, jacobiCudaGraphs, shmembench, cudaTensorCoreGemm: shared-memory, atomic and tensor bound, so the share rule declines them and they run the plain loop): 0.998, 1.005,
  0.996, 1.006 (geometric mean 1.001). radixSortThrust (15 % computed by generated code, 0.993), fpdc-cuda (0.982, spreads of 34 % and 21 %: it writes 800 MB a run) and black-scholes-cuda (0.984, spreads of 7 % and 12 %: 18 lanes active on average, so
  the branches split the group) are inside their own noise. histogram gains 11 % although generated code is 3.9 % of its profile and about 65 % of it is the shared-memory and atomic handlers (`perf`, below): where the rest of the gain comes from (the loop around the handlers: the dispatch of the ops that run in regions) was not separated.
* **What stops it.** What is left is what the generated code still calls: the loads, stores, shared-memory ops and atomics (36 % of the time of these programs by the study's class table: 15.3 % global loads, 12.2 % shared, 6.3 % global stores, 2 % atomics),
  the branches that split a group, BSSY/BSYNC/EXIT/barriers/shuffles that end a region, fp64 and the SFU. SYRK is the clean example: 194 M instructions, 47 % computed by generated code at about 2 ns each instead of 8, 100 M escaped loads and stores. By the study's per-op costs (server2) the tier's 4.1 s are 2.4 + 1.0 s in those escapes and 0.7 s in the ALU ops: 83 % that no
  instruction form touches. Making every ALU op free would give 1.2x at most; 1.125 is what 2 ns an op gives. Getting from 1.10 to 1.15 needs the memory ops and shared memory inside generated code with their exact counters (the study's extensions (a) and (c)), which is a different design problem.
* **What it costs.** A unit that is compiled for the first time costs 2.3 s on average (67 units, 155 s for the 22 programs; base: 37 units, 62 s), because regions are bigger and each loop is one function: the cold run is 1.22x the tier's wall on geometric mean (base 1.14x),
  SYRK 1.75x, segmentationTreeThrust 3.2x (22 units). All of the cold penalty is the compile: without it the same runs are 0.92x of the tier, as the warm ones (0.91x). `VGPU_JIT_RESIDENCE=0 VGPU_JIT_SPECIALIZE=0` cuts the compile by 22 to 48 % and
  loses nothing at run time (2 % faster in geometric mean over 7 programs, 3 runs each); it was found too late to become the default of this round (every correctness result below is for the full form), and **became the default in round sp38**, whose matrix (the seven `e2e_sass_jit` shards, `test_jit*`, the 22-program same-bits comparison at 1 and 4 threads, the warm and cold tables with an A/A) was run again for it: [docs/performance.md](../performance.md), "Round sp38 (follow-ups)", `data/sp38/`. Everything below that says "default" about the generated code's form is the full form, now `VGPU_JIT_RESIDENCE=1 VGPU_JIT_SPECIALIZE=1`.
* **Same bits.** Results, every counter and `VGPU_KERNEL_DIGEST` equal to the tier-only run and to `origin/main` on all 22 programs at one thread (cold and warm cache) and four threads, except where the tier-only run of `origin/main` already differs from itself (radixSortThrust at one and four threads,
  jacobiCudaGraphs and segmentationTreeThrust at four: controls below).


## Why the generated code covered half of what ran in regions (the study's open question)

The study's table 1 has two columns that look like a contradiction: SYRK 81.1 % of its instructions "ran in regions" and 45.5 % were "computed by generated code" (SYR2K
90.4 and 56.8). It is not a contradiction and nothing was lost. "Ran in regions" counts the **escapes**: an LDG or STG inside a region is a call of the tier's own handler, and it
is counted as a region instruction. SYRK is 51.8 % loads and stores; its regions held 67 M escaped loads (the JIT statistics: 157.6 M instructions ran in 33.6 M calls, 88.4 M of them
computed, so 69 M were escapes) and the other 19 % were the stores (a store could not be in a region) and the branches, S2R and uniform ops, which ended the regions. The generated coverage is bounded by the opcode set
(45.7 % "JIT-form ceiling" in the study), not by the region extents. What the regions did cost was the *call*: 33.6 M calls of about 5 instructions, 23 ns each, and an escape that
was 2.6 ns dearer than in the tier's loop. That is what this round removes.

## What was built

### 1. A region is a run of ops the run loop goes on past, cut into stretches

The tier's run loop already has a rule for going on past an op without leaving (the "cont path", `docs/performance.md`, branch `sp14-step`): when the chain would have started the
next call exactly where the loop is, the loop does what the chain does -- `--w.budget; ++w.extra`, the 256-op bound starts afresh -- and carries on. It does it for a BRA that the
whole group takes, or none of it, and for S2R, ULDC, UMOV, USEL, PLOP3 and SEL. The generated code now does exactly that, inside one function:

| the loop's rule (`tier_run_t`) | where it is in generated code | mutant (`VGPU_JIT_MUTANT=`) |
| --- | --- | --- |
| only when the chain conditions hold (`regs_ok`, no statistics, `tier_chain_`, no yield / slice / give-way / WARPSYNC hold / forced mask, `runnable() == group`), worked out once per pass | the host works them out once per pass and hands them in as `kFlagChain`; a branch or cont op that finds it clear leaves before doing anything | `ctl_flag`, `host_chain` |
| and the turn has a call left (`w.budget`) | `*c->budget == 0` leaves; taking a branch does `--budget; ++extra` | `ctl_budget`, `ctl_budget_dec`, `ctl_extra` |
| a branch is taken by the whole group or by none of it | `take != group && take != 0` leaves, the branch uncounted (the loop then runs it: divergence, `select_pcs`, the spin rule) | `ctl_split`, `ctl_swap` |
| the lanes the guard leaves out, a branch with two condition predicates, the divergent-branch counter | `ex`, `take &= P`, `++*c->divergent` when `ex && ex != group` | `ctl_divcount` |
| the 256-op run bound counts straight ops only, restarts at every op the loop goes on past | `n0 + straight > 256` leaves at a stretch's start, `n0 = 0` after each branch or cont op; the host tracks where the run restarted (`exit_ps`) and sets its own bound from it | `ctl_n0check`, `ctl_n0`, `ctl_ps`, `host_ps`, `host_n0` |
| the step budget: an op that would be the `max_steps + 1`-th is the loop's to throw | `*steps + stretch > max_steps` leaves at the stretch's start | `ctl_stepscheck` |
| a cont op ends the stretch like a branch, and is counted and run as the loop does | the same gate before it, then its handler (an escape) or its generated body | `ctl_cont_nogate` |
| the store gate: a warp's first STG or ST ends its turn, once | `*c->gate` leaves before a store (whatever its guard says); once the gate is down a store is an op like another, so **a store no longer ends a region** | `ctl_gate` |
| the number of calls a turn is made of, the clock readings (S2R of SR_CLOCK counts `w.steps`) | an S2R of a clock stays an escape (counters flushed before it); every other counter is written before any call and before the function leaves | -- |

A region is a run of consecutive such ops (up to `VGPU_JIT_MAX_REGION`, default 200 now: the stretches are what the 256 bound is about). It is cut into **stretches**: a stretch
starts at the region's first op, at the target of a branch inside the region, and at the op after a branch or cont op, and ends at the next branch or cont op. Every stretch
start is an entry (`Unit::region_at`), so a warp that comes back to a loop's top from the tier finds the compiled loop there. A branch to an op outside the region leaves with
that op as the next. A taken backward branch is a `goto`: a loop is one call of the generated function for as many iterations as the turn's budget allows.

What leaves is always "before the op, nothing of it done": the loop then runs that op itself, with the counters, `w.pc` and the pass state as it would have had them. The host side (`jit_run_region`)
takes the next op and the op at which the 256-op run restarted from the function and carries on; if the function left at once it says "nothing ran" and the op goes through
the loop as before.

### 2. More ops computed by generated code (they were escapes or ended regions)

`SEL`, `PLOP3`, `S2R` (not the clocks; the values come from the host through `Ctx::sreg`, the tier's `tier_sreg_fill`), `CS2R` of SRZ, `UMOV`, `ULDC` (bank 0), `UIADD3`, `ULOP3`,
`USEL`, `UISETP` (with uniform predicate results). The rest of the uniform datapath (`UIADD3`, `UIMAD`, `UISETP`, `ULEA`, `ULOP3`, `USHF`, `UFLO`/`UPOPC`) and `CS2R` are not on the tier's
`cont` list; the JIT treats them as the same kind of op (a handler that only writes registers) inside a region, and the plain loop is unchanged. Each form follows its
handler (named in `jit_gen.cpp`) and has a mutant (`VGPU_JIT_MUTANT=<form name>`).

### 3. The host side

* `tier_run_t<true>` goes on past branches and cont ops itself, as the plain loop does (it did not: every branch ended the pass), and issues plain register-only runs back to back
  (`Unit::jrun`, a run that stops short of the next entry of a region). That was part of the study's "15 ns per region call in the run loop".
* The loop reads the entries of the compiled regions only (`Unit::active`, switched when the unit is Ready): a region that was not selected costs no call. The function
  pointers are a flat array (`Unit::fast`).
* An escape calls the tier's handler directly (`Ctx::ops`, `Ctx::tctx`; the handler is the first word of the 96-byte `MOp`) instead of through a callback that called it.
* The compile command drops the distribution's stack protector, stack-clash probes and control-flow instrumentation (`-fno-stack-protector -fno-stack-clash-protection -fcf-protection=none`), a region entered at its first op only has no jump table, and the generator can emit one body per region (`VGPU_JIT_SPECIALIZE=0`; since round sp38 this is the default, `1` emits the second body for whole warps).
* New knobs: `VGPU_JIT_MIN_GEN` (the share of a region's ops that must be generated, default 50: the old fixed rule), `VGPU_JIT_SPECIALIZE`, `VGPU_JIT_DUMP=<file>` (a listing of every section that gets regions, op by op, with G / E / B / C marks); `VGPU_JIT_MAX_REGION` defaults to 200 (was 96); the statistics line says how many branches and register-only ops the generated code went on past.
* `Ctx` ABI 3 (`kAbiVersion` 3, `kGeneratorVersion` 4): `up` and `ur` are writable, new fields `divergent`, `gate`, `budget`, `extra`, `gen_run`, `sreg`, `max_steps`, `tctx`, `ops`, `last`, `exit_idx`, `exit_ps`.

## Measurements

Protocol (the study's, with two changes): `VGPU_THREADS=1`, `taskset -c 9`, one process at a time, **warm cache** = `VGPU_JIT=1 VGPU_JIT_THRESHOLD=100000 VGPU_JIT_SYNC=1` on a private cache directory filled by an untimed first run (the threshold is 1000x below the default so that every hot section compiles; the share rule,
`MIN_REGION`, `HOT_PERMILLE` are the defaults), **5 interleaved runs per side** (A,B then B,A, ...), medians of whole-process wall, spread = (max - min) / median; builds are Release with LTO on server3 (Ryzen 7 PRO 8840U, 8 cores / 16 threads, g++ 13.3). The tier-only side is the same binary
without `VGPU_JIT`. The machine was shared with other sessions (a lock serialised every timed run; the 1-minute load average gate was 9 for these runs; other sessions' builds still ran at times, which is why a few spreads are large). Each run is one process; every
side of a program runs inside one hold of the lock. Not done: AVX-512 object choice, the default threshold of 100 M (every number here uses 100 000), a second machine. Scripts: `data/round15/scripts/`, raw data `data/round15/*.json`.

### Warm cache, all 22 programs (`data/round15/final_warm.json`, `final_warm2.json`, `aa.json`)

A/A' is a second tier-only side in the same interleaving (5 programs in the first set, 10 more in `aa.json`: 0.987 to 1.011 over 15, listed after the table).

| program | tier A s (spread) | tier again A' s | JIT s (spread) | **A/JIT** | A/A' | warp instructions M | ran in regions % | computed by generated code % | region calls M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| polybench/SYRK | 5.40 (1%) | 5.34 | 4.80 (3%) | **1.125** | 1.010 | 194 | 100 | 47 | 0.10 |
| polybench/SYR2K | 10.20 (5%) | 10.32 | 8.97 (4%) | **1.136** | 0.988 | 399 | 100 | 57 | 0.10 |
| samples/convolutionSeparable | 11.39 (2%) | 11.55 | 11.41 (21%) | **0.998** | 0.987 | 385 | 0 | 0 | 0 |
| samples/mergeSort | 8.50 (2%) | 8.50 | 7.80 (2%) | **1.090** | 1.000 | 354 | 73 | 53 | 20.39 |
| samples/sortingNetworks | 13.67 (0%) | 13.66 | 12.90 (1%) | **1.060** | 1.001 | 492 | 44 | 33 | 8.26 |
| hecbench/fpdc-cuda | 236.96 (34%) | - | 241.25 (21%) | **0.982** | - | 258 | 67 | 52 | 18.75 |
| samples/interval | 15.87 (3%) | - | 14.54 (1%) | **1.091** | - | 795 | 81 | 49 | 36.13 |
| samples/histogram | 13.70 (1%) | - | 12.37 (1%) | **1.107** | - | 448 | 80 | 61 | 9.29 |
| hecbench/black-scholes-cuda | 20.16 (7%) | - | 20.50 (12%) | **0.984** | - | 1222 | 75 | 65 | 115.61 |
| samples/jacobiCudaGraphs | 19.03 (1%) | - | 18.94 (1%) | **1.005** | - | 916 | 0 | 0 | 0 |
| polybench/LU | 29.91 (1%) | - | 25.34 (3%) | **1.180** | - | 1777 | 90 | 82 | 136.62 |
| samples/simpleHyperQ | 11.94 (1%) | - | 7.02 (2%) | **1.701** | - | 1176 | 100 | 40 | 0.11 |
| samples/alignedTypes | 50.19 (2%) | - | 45.86 (2%) | **1.094** | - | 972 | 72 | 45 | 0.50 |
| samples/radixSortThrust | 39.52 (1%) | - | 39.79 (0%) | **0.993** | - | 1387 | 18 | 15 | 32.33 |
| polybench/FDTD-2D | 99.69 (1%) | - | 85.45 (2%) | **1.167** | - | 5373 | 89 | 74 | 393.19 |
| hecbench/shmembench-cuda | 4.73 (1%) | - | 4.75 (2%) | **0.996** | - | 186 | 0 | 0 | 0 |
| samples/reduction | 9.54 (1%) | - | 7.61 (1%) | **1.253** | - | 348 | 100 | 77 | 0.15 |
| samples/transpose | 8.18 (4%) | - | 7.01 (5%) | **1.166** | - | 427 | 80 | 68 | 13.21 |
| samples/segmentationTreeThrust | 33.05 (20%) | - | 30.23 (12%) | **1.093** | - | 610 | 64 | 56 | 61.41 |
| samples/cudaTensorCoreGemm | 13.55 (5%) | - | 13.47 (0%) | **1.006** | - | 59 | 0 | 0 | 0 |
| hecbench/atomicAggregate-cuda | 4.73 (2%) | - | 4.35 (3%) | **1.088** | - | 164 | 78 | 60 | 14.67 |
| rodinia/gaussian | 3.67 (1%) | - | 3.53 (2%) | **1.039** | - | 165 | 89 | 84 | 15.45 |

programs 22; geomean A/JIT **1.0989**; worst 0.982 (hecbench/fpdc-cuda); best 1.701 (samples/simpleHyperQ); slower than -2%: none; sum A 664 s, sum JIT 628 s (1.057x)
A/A' over 5 programs: 0.987 to 1.010 (geomean 0.9971)
output hashes: DIFFER in samples/segmentationTreeThrust

A/A' of the 15 programs that have one (tier against the same tier, same protocol): atomicAggregate 1.001, shmembench 1.001, SYR2K 0.988, SYRK 1.010, gaussian 0.999, convolutionSeparable 0.987, cudaTensorCoreGemm 1.011, histogram 1.003, interval 0.992, jacobiCudaGraphs 1.000, mergeSort 1.000, reduction 1.010,
simpleHyperQ 0.997, sortingNetworks 1.001, transpose 0.998 (geometric mean 0.9998). The 7 programs without one (fpdc, black-scholes, LU, alignedTypes, radixSort, FDTD-2D, segmentationTreeThrust) are the long or the I/O-bound ones; their spreads are in the table.
In the timed runs no JIT run compiled (`compiles` is 0 in every sample) and none failed. The "output hashes" line is the harness's normalised-output check; segmentationTreeThrust's hashes differ between the sides only because the JIT's statistics (`VGPU_JIT_STATS=1`) print a
line in the middle of one of the program's own output lines ("Building segmentation tree... jit: share rule ... done in ... ms"); the same bits check without the statistics (below) is exact.

### Against today's JIT (the exploration run, v2 build, 3 runs; the final build changed only the compile flags, the entry dispatch and a host-side flags word, and is measured above for the tier only)

| program | tier s | base JIT s (A/B) | this branch (v2 build) s (A/B) | A/B base | **A/B v2** |
|---|---:|---:|---:|---:|---:|
| polybench/SYRK | 5.44 | 5.72 | 4.78 | 0.951 | **1.138** |
| polybench/LU | 30.39 | 25.94 | 25.41 | 1.171 | **1.196** |
| rodinia/gaussian | 3.63 | 3.57 | 3.46 | 1.017 | **1.049** |
| samples/transpose | 8.11 | 7.05 | 6.87 | 1.151 | **1.180** |
| samples/simpleHyperQ | 11.87 | 12.01 | 6.93 | 0.989 | **1.712** |
| samples/histogram | 13.55 | 12.04 | 12.40 | 1.126 | **1.093** |
| samples/reduction | 9.42 | 8.10 | 7.64 | 1.163 | **1.233** |
| polybench/SYR2K | 10.15 | 10.51 | 9.12 | 0.966 | **1.114** |
| samples/mergeSort | 8.43 | 8.32 | 7.77 | 1.014 | **1.084** |

geomean over these 9 programs: base 1.057, this branch (v2 build) 1.187

### Same bits (`verify.py`; `data/round15/verify/`)

Reference: `origin/main` 7d8d98a4 (the base build), tier only. Variants: this branch tier only, this branch with the JIT forced on at once (threshold 100 000, synchronous), **cold** cache and then **warm**. Compared: the normalised output, `VGPU_COUNTERS_FILE` (every launch, without the pid)
and `VGPU_KERNEL_DIGEST`, and the exit status. 22 programs x {1, 4} threads x 3 variants = 132 comparisons; 120 are equal. The other 12 are four programs at one thread count each where **the reference differs from itself**: `radixSortThrust` (counters and digest, at 1 and 4 threads: its look-back
launches poll, and a failed poll is an instruction), `jacobiCudaGraphs` (digest, 4 threads), `segmentationTreeThrust` (counters, 4 threads). Controls: the same comparison of `origin/main`'s tier-only run against itself gives DIFFERENT for exactly these programs and thread counts
(`verify/verify_control.log`, `verify_control2.log`), with equal output; in these three the variants' output is equal to the reference's and the counters or the digest that differ are the ones that differ between two runs of the reference.

### Cold cache (the first run, empty cache directory, compile on the launching thread; `data/round15/final_cold.json`, `cold_v4.json`)

| program | tier s (warm table) | base JIT, cold: s (compiles, ms) | this branch, cold: s (compiles, ms) | cold / tier base | cold / tier this |
|---|---:|---|---|---:|---:|
| polybench/SYRK | 5.40 | 6.95 (1, 1113) | 9.43 (1, 4577) | 1.29 | 1.75 |
| polybench/SYR2K | 10.20 | 12.34 (1, 1460) | 11.91 (1, 2414) | 1.21 | 1.17 |
| samples/convolutionSeparable | 11.39 | 11.33 (0, 0) | 11.30 (0, 0) | 0.99 | 0.99 |
| samples/mergeSort | 8.50 | 11.95 (3, 3501) | 16.98 (4, 8058) | 1.41 | 2.00 |
| samples/sortingNetworks | 13.67 | 13.71 (0, 0) | 15.86 (3, 3113) | 1.00 | 1.16 |
| hecbench/fpdc-cuda | 236.96 | 281.40 (1, 1700) | 175.92 (1, 3125) | 1.19 | 0.74 |
| samples/interval | 15.87 | 16.95 (1, 738) | 21.98 (3, 7078) | 1.07 | 1.39 |
| samples/histogram | 13.70 | 16.68 (3, 4492) | 18.91 (3, 6484) | 1.22 | 1.38 |
| hecbench/black-scholes-cuda | 20.16 | 23.87 (1, 3156) | 27.54 (1, 7106) | 1.18 | 1.37 |
| samples/jacobiCudaGraphs | 19.03 | 19.36 (0, 0) | 18.89 (0, 0) | 1.02 | 0.99 |
| polybench/LU | 29.91 | 27.13 (2, 1682) | 28.41 (2, 3182) | 0.91 | 0.95 |
| samples/simpleHyperQ | 11.94 | 11.95 (0, 0) | 7.71 (1, 620) | 1.00 | 0.65 |
| samples/alignedTypes | 50.19 | 50.73 (3, 3880) | 59.28 (8, 11480) | 1.01 | 1.18 |
| samples/radixSortThrust | 39.52 | 47.24 (1, 8193) | 45.92 (1, 6067) | 1.20 | 1.16 |
| polybench/FDTD-2D | 99.69 | 90.14 (2, 1428) | 88.64 (3, 2633) | 0.90 | 0.89 |
| hecbench/shmembench-cuda | 4.73 | 4.81 (0, 0) | 4.76 (0, 0) | 1.02 | 1.01 |
| samples/reduction | 9.54 | 9.88 (2, 1485) | 9.57 (2, 1737) | 1.04 | 1.00 |
| samples/transpose | 8.18 | 11.58 (6, 4466) | 14.87 (8, 7561) | 1.42 | 1.82 |
| samples/segmentationTreeThrust | 33.05 | 54.67 (8, 22038) | 106.73 (22, 76046) | 1.65 | 3.23 |
| samples/cudaTensorCoreGemm | 13.55 | 13.36 (0, 0) | 13.66 (0, 0) | 0.99 | 1.01 |
| hecbench/atomicAggregate-cuda | 4.73 | 5.93 (1, 1161) | 5.26 (1, 986) | 1.25 | 1.11 |
| rodinia/gaussian | 3.67 | 5.55 (1, 2016) | 6.52 (2, 2940) | 1.51 | 1.78 |

compiles: base 37 (62.5 s), this branch 67 (155.2 s); geomean of cold wall over tier wall: base 1.142, this branch 1.220

(fpdc-cuda writes 800 MB a run: its cold row is the machine's disk, not the JIT.) The cold run is the tier's run plus the compile, to within the noise: without the compile time the 21 programs without fpdc are 0.921x of the tier (cold) against 0.905x (warm).

### Cold cost and what to do about it (`data/round15/cold_variants.json`, `resid.json`)

The compile is what the new generator made dearer. `-ftime-report` on SYRK's unit (207 ops, 148 KB of C++, 142,844 bytes of machine code): 3.4 to 3.9 s wall, 345 MB, parsing 1.6 s (the fixed part, `#include`ing the lane kernels, is 0.42 s), optimisation and code generation 3.2 s;
one body per region instead of two (`VGPU_JIT_SPECIALIZE=0`): 2.5 s; `-O1`: 1.5 s. Two knobs that already exist or were added for this, cold, one run each:

| program | default | `VGPU_JIT_RESIDENCE=0` | `VGPU_JIT_SPECIALIZE=0` | both |
|---|---|---|---|---|
| polybench/SYRK | 9.5 s wall, 1 compiles, 4630 ms | 8.6 s wall, 1 compiles, 3829 ms | 8.3 s wall, 1 compiles, 3105 ms | 7.6 s wall, 1 compiles, 2397 ms |
| polybench/LU | 28.4 s wall, 2 compiles, 2804 ms | 26.9 s wall, 2 compiles, 2285 ms | 30.0 s wall, 2 compiles, 2384 ms | 27.1 s wall, 2 compiles, 2076 ms |
| samples/reduction | 9.5 s wall, 2 compiles, 1717 ms | 9.2 s wall, 2 compiles, 1560 ms | 9.4 s wall, 2 compiles, 1608 ms | 9.0 s wall, 2 compiles, 1315 ms |
| samples/transpose | 15.1 s wall, 8 compiles, 7868 ms | 14.1 s wall, 8 compiles, 7117 ms | 14.3 s wall, 8 compiles, 6792 ms | 13.3 s wall, 8 compiles, 6152 ms |
| samples/mergeSort | 15.6 s wall, 4 compiles, 7797 ms | 14.5 s wall, 4 compiles, 6985 ms | 14.4 s wall, 4 compiles, 6377 ms | 13.4 s wall, 4 compiles, 5681 ms |

and the run time they give (warm, 3 runs; "residence" keeps the rows an op touches in local arrays within a stretch, as before; without it the generated code reads and writes the register file):

| program | tier s | JIT (rows in locals) s | JIT, `VGPU_JIT_RESIDENCE=0` (rows in memory) s | A/JIT | A/JIT residence off |
|---|---:|---:|---:|---:|---:|
| polybench/SYRK | 5.35 | 4.83 | 4.76 | 1.109 | 1.125 |
| polybench/LU | 29.83 | 25.23 | 24.61 | 1.182 | 1.212 |
| samples/reduction | 9.46 | 7.59 | 7.43 | 1.247 | 1.274 |
| samples/transpose | 8.06 | 7.00 | 7.00 | 1.152 | 1.152 |
| rodinia/gaussian | 3.63 | 3.56 | 3.37 | 1.018 | 1.077 |
| samples/simpleHyperQ | 12.05 | 6.97 | 6.74 | 1.728 | 1.788 |
| samples/mergeSort | 8.50 | 7.77 | 7.75 | 1.094 | 1.097 |

geomean over these 7 programs: 1.202 with residence, 1.229 without

Residence does not pay in these programs (their regions are mostly escapes and short stretches), and it is the part of the generated code that a copy-and-patch JIT could not have ([the study](copy-and-patch.md)). `VGPU_JIT_RESIDENCE=0 VGPU_JIT_SPECIALIZE=0` is the better default;
it was not made one in this round because every correctness result in this document is for the full form; round sp38 re-ran the matrix (`e2e_sass_jit` shards, the verify above, `test_jit*`) and made it the default (`VGPU_JIT_RESIDENCE=1` and `VGPU_JIT_SPECIALIZE=1` are the knobs for the full form).

### Where the time is (`perf record`, histogram, warm cache; base and this branch look the same)

18.7 % `bank_passes_bitmap`, 18.0 % `exec_mem` (the atomics), 11.4 + 7.0 % `h_shared`, 5.8 % `tier_run_jit`, 5.4 % `shared_lanes`, 4.6 % `sass_bank_passes_widened`, generated code (`vgpu_jit_r*`) 3.9 %, `jit_run_region` 1.07 % (base build: 0.57 % of 12.0 s, 11.5 M calls against 9.3 M: about 14 ns a call in the v2 build against 6 ns in base; the final build's changes were meant to lower the 14 and are not profiled). The gain it has is therefore not in the generated code itself, and more ALU forms would not add to it.

### Per-call cost, measured and not

The study fitted 8.5 ns in generated code and about 15 ns in the loop per region call. Here the host side of a call is `jit_run_region` (about 14 ns in the histogram profile: 1.1 % of 12.4 s over 9.3 M calls) and the generated function's own prologue; LU makes 136.6 M calls and spends 1.1 s in `jit_run_region`. The generated prologue
carried a stack protector, stack-clash probes and an `endbr64`/`notrack` entry switch (Ubuntu's defaults); the compile command now turns them off and a region entered at its first op only has no jump table. I did not isolate those two changes in a timing of their own (the build with them is the only one measured after
the exploration run), so what they gave is not a measured number.

## Caveats and what is not solid

* One machine, shared; timings are 5-run medians (3 in the exploration and variant tables). The spreads are in the tables; an effect inside the A/A range (0.987 to 1.011) is not an effect. The programs fpdc-cuda, black-scholes-cuda and segmentationTreeThrust have spreads of 7 % to 34 % at times.
* The geometric mean uses the tier of the same binary as the denominator (the base build's tier differs from this branch's by layout: +-3 % on programs the JIT does not touch, as round 14 found).
* The per-call figures above come from one profile (histogram) and from the arithmetic of two runs (LU); they carry the sampler's error.
* The mutants that survive in e2e (table above) are caught by `test_jit_resid`, except `host_ps` and `host_maxreg` (host-side bookkeeping that changes how many calls a turn is made of, or that guards a case no program here reaches): no test in the tree fails with them.
* The new forms `CS2R` (SRZ only) and the uniform ones are covered at run time by segmentationTreeThrust, gaussian and jit_loops, not by a hand-written test; a hand-written SASS test of each uniform form (as `test_sass_chain_cont` has for S2R) would be the better holder and was not written.
* `VGPU_JIT_STATS=1` with the tier's statistics on (`VGPU_TIER_STATS`) stops the generated code at every branch and cont op (the loop's chain is off then), so the "branches continued" statistic reads 0 in such a run; the gate checks it in a run without them.


## Tests (all CMake-wired; the mutation evidence is in "Mutation checks")

| test | what it holds the generated code to |
| --- | --- |
| `test_jit_resid`: `jit_graph_regions_equal_the_model_of_the_run_loop` (new) | random regions of escapes, stores, branches (guarded, one or two condition predicates, targets inside and outside the region, backward) and cont ops, 12 per unit, 40 trials each over a group (whole warp, one lane, two lanes, half, random), an entry (any stretch start), the run bound already taken (0 to 300), the turn's calls left (0 to 200), the step budget (close to the limit one time in four), the chain flag and the store gate, against a model of the run loop written from `tier_run_t`'s rules (not from the generator): counters, registers, predicates, `budget`, `extra`, the divergent-branch count, where it leaves and where the 256-op run restarted, and an escape that throws (counters then) |
| `jit_graph_regions_with_residence_equal_the_register_file_code` (new) | the same kind of regions with the generated forms among them (the new ones too), compiled with and without register residence: every row, predicate, uniform register and predicate, counter and exit must agree (residence across labels, branches, cont ops and stores) |
| `jit_residence_random_regions_equal_the_register_file_code` | as before, with the ten new forms in the random mix |
| `test_jit`: regions, entries, jrun, `stretch_starts`/`stretch_end` | discovery: a store no longer ends a region, a branch to an odd offset / past the section / backward out of it / with three conditions is not eligible, stretch starts are the first op, branch targets inside and the op after a branch or cont op, and the calls through `Ctx::ops` |
| `test_sass_chain_cont_jit`, `test_sass_store_gate_jit`, `test_sass_alu_runs_jit` (new registrations) | the existing hand-written SASS tests (words, counters, clocks, the store gate against the interpreter) with every region compiled at once |
| `e2e_sass_jit_shard_loops` (new shard, `nvidia/tests/e2e/jit_loops.cu`) | nine kernels with loops and branches (see the file), five block shapes (32, 64, 96 and 33 threads, one thread), results and every counter equal to the tier at one and four threads, with a warm cache, in the background, without the chain, with the share rule on; and the generated code must have gone on past at least 1000 branches or cont ops (`VGPU_JIT_MIN_CONTINUED`) |
| the other `e2e_sass_jit` shards (`tier`, `steps`, `steps_gather`, `steps_shared`, `random`, `modes`) | unchanged tests, now running the new code: sass_tier's 30 units, sass_tier_steps' faults (at least 2 thrown inside a generated region), the two step-limit bisections, the random kernels |

**Results on the measured build** (server3, no nvcc there: the programs were compiled with nvcc 13.0 on another machine and run through `run_sass_jit.sh` with `VGPU_TIER_WORK`): all seven `e2e_sass_jit` shards pass (`data/round15/verify/gate5.log`: loops, tier, steps with 2 faults thrown inside generated regions, random, modes, steps_gather, steps_shared); `test_jit` 24/24, `test_jit_share` 6/6, `test_jit_resid` 3/3;
`test_sass_chain_cont_jit`, `test_sass_store_gate_jit`, `test_sass_alu_runs_jit`, `test_tier` and the three plain ones pass (an earlier build, `data/round15/verify/ctest_jit.log`; the tests do not depend on what changed after). Not run: the sanitizer builds (the JIT is off in them), CI itself (nothing was pushed).

`nvidia/tests/e2e/run_sass_jit.sh` can now run on a rig without nvcc with programs built beforehand (`VGPU_TIER_WORK`, `jit_random.<seed>.<arch>`).

## Mutation checks (`-DVGPU_JIT_MUTANTS=ON`, `VGPU_JIT_MUTANT=<name>`; logs in `data/round15/`)

A test that does not fail with a mutant does not test the rule. Every mutant below is a switch in the generator (`jit.cpp`, `jit_gen.cpp`) or in the executor's side (`tier_ops.inc` through `jit::host_mutants()`), compiled in
only with `-DVGPU_JIT_MUTANTS=ON`. Unit tests: `test_jit_resid` (3 units, 3 units of graph regions, 3 CPUs; pristine passes 3/3). e2e: the JIT with every region compiled (`VGPU_JIT_THRESHOLD=1 VGPU_JIT_HOT_PERMILLE=0`)
against the tier on a prebuilt program, outputs and every counter compared; "caught" is any difference.

| mutant | what it breaks | `test_jit_resid` (model test) | e2e |
| --- | --- | --- | --- |
| `ctl_flag` | goes on past a branch whatever the chain conditions say | FAIL | caught (jit_loops) |
| `ctl_budget` | ignores that the turn has no call left | FAIL | caught (jit_loops, race kernels) |
| `ctl_budget_dec` | does not charge the turn for a branch it goes on past | FAIL | caught (jit_loops) |
| `ctl_extra` | does not tell the turn how many calls it took over | FAIL | caught (jit_loops) |
| `ctl_split` | runs a branch that splits the group | FAIL | caught (jit_loops) |
| `ctl_swap` | takes the branch the lanes did not | FAIL | caught (jit_loops) |
| `ctl_nocount_bra` | does not count the branch | FAIL | caught (jit_loops) |
| `ctl_divcount` | does not count a divergent branch | FAIL | survives (the counter is not in the programs' output) |
| `ctl_n0check` | no check of the 256-op run bound at a stretch's start | FAIL | caught (jit_loops, race2) |
| `ctl_n0` | the run bound is not restarted by a branch | FAIL | survives (see below) |
| `ctl_ps` | does not say where the run restarted | FAIL | survives (see below) |
| `ctl_stepscheck` | no check of the step budget at a stretch's start | FAIL | survives in jit_loops (the step-limit stage of `e2e_sass_jit_shard_steps` is its test) |
| `ctl_cont_nogate` | goes on past a cont op without the gate | FAIL | survives (see below) |
| `ctl_gate` | runs a store while the warp's store gate is up | FAIL | caught (jit_loops, race kernels) |
| `host_chain` | the host says the chain conditions hold when they do not | (host code: not in this test) | caught (jit_loops) |
| `host_n0` | the host hands in 0 as the ops the pass has taken | -- | caught (jit_loops) |
| `host_ps` | the host forgets where the run restarted | -- | survives (changes the calls a turn is made of only where an op is run by the loop and by a region alternately, never the results of these programs) |
| `host_maxreg` | no check that the warp's register file holds what the region touches | -- | survives (no program here resizes a warp's file under a running region) |

Forms (the generated code for a form made wrong in one lane): the 19 forms of the earlier rounds and the new ones are all caught by `e2e` (sass_tier for the 19 of the earlier rounds, `sel`, `plop3`, `umov`, `s2r`; the uniform
and CS2R forms need a program that uses them: jit_loops (`s2r`, `uiadd3`, `umov`, `sel`, `plop3`), segmentationTreeThrust (`uiadd3`, `uldc`, `ulop3`, `cs2r`, `uisetp`, `plop3`, `sel`, `s2r`, `umov`), gaussian (`uldc`, `uiadd3`, `ulop3`, `s2r`)),
and `usel` (segmentationTreeThrust). In sass_tier itself `uldc`, `uiadd3`, `ulop3`, `usel`, `uisetp` and `cs2r` survive: its kernels use none of them where the result is observed (found by this check; the programs above were added as targets for that reason). Two mutants were wrong when first run, which the check also showed: `s2r` corrupted a lane twice (its own flip and `store()`'s) and so survived, `usel` had no corruption at all; both were fixed and re-run (`data/round15/mutation/e2emut_forms3.log`, `e2emut_forms7.log`). gaussian's `umov` survives (it has no UMOV that is observed), and the unmutated control (`none`) is identical to the tier in every target.


