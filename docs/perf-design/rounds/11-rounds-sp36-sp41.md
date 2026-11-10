## Round sp36 (memory counting): the passes and sectors of the warps that the closed forms and the table of shapes miss (branch `sp36-counting`, base `origin/main` 735c2626)

Question: pooled over 15 real NVIDIA programs (class study of 2026-11-06) global memory is 22.5 % of the cycles at 42.6 ns a warp instruction and shared memory 10.6 % at 44 ns; about a fifth of an instruction's cost is the handler, the rest the memory manager and the exact counters. Make the counting cheaper without moving one counter: where does it go, per access pattern, and which patterns are still counted by general code?

Same-bits rule: `VGPU_COUNTERS_FILE`, `VGPU_KERNEL_DIGEST` and the program's output are identical to `origin/main` at 1 and 4 threads for the nine programs timed (table below). Hosts: server2 (Zen 5, Ryzen 9 9950X, `nice -n 19`, `-j3`, one builder under `flock`) for timing, tests and mutants; server3 (Zen 4 mobile, `perf`) for profiles. Builds are Release/LTO with `-fno-omit-frame-pointer -g1` on both sides (the same flags for A, F and the controls).

### Where the time went (A = 735c2626, per access pattern)
Method: an instrumented build of A (`rdtsc` around each counting entry point, a class set by the path that answered; 1 thread, cpu 24; the cycles are TSC cycles of one call and include the ~31 cycles an `rdtsc` pair costs; nothing like it is in the tree), then `perf record -e cycles:u` of the same programs on server3. The programs' counting paths and what the warps looked like (a few warps of each pattern were dumped):

| program | counting, share of cycles (perf, A) | pattern, and the path that counted it |
|---|---:|---|
| histogram | 24 % | 44.3 M shared warps of 32 byte lanes on random bytes over 1 to 4 KB (`bank_passes_bitmap` 15.8 %, `widened` 4.0 %, table 1.3 %, the kept-warp check 1.2 %): 282 cycles a miss, 35.6 M more answered by the warp kept from the load of a load/store pair (36 cycles) |
| radixSortThrust | 22 % | 26.1 M shared warps of 4-byte lanes on random words over 32 KB and 25.9 M of 8-byte lanes on one or two words, all through the table of words (`sass_bank_passes_table` 16.9 %, 633 cycles a call: a 16 KB frame, a bank's words 512 bytes from the next bank's); 25.8 M global warps of two or three runs of consecutive words (`sectors_monotone`, 169 cycles) |
| sortingNetworks | 26 % | 112 M shared warps found in the table of shapes (`sass_bank_passes_offsets` 12.9 %: 36 cycles at the entry the instruction used last, 49 elsewhere), 13.6 M bitonic partners no table held (bitmap 7.1 %, widened 4.7 %) |
| mergeSort | 27 % | 17.8 M misses (data-dependent windows: bitmap 16.5 %, widened 3.6 %), 9.5 M table hits, 5.6 M strides |
| convolutionSeparable | 15 % | 139 M shared hits (`sass_bank_passes_offsets` 9.6 %), 22.5 M global warps of two rows of 16 words 12,288 bytes apart (`sectors_monotone`, 122 cycles, 4.5 %) |
| alignedTypes, FDTD-2D, SYR2K, shmembench | under 2 % | a coalesced run (`count_run`: 225 M, 917 M, 34 M calls) or a full-warp stride (`count_global_progression`: 175 M, 134 M) is counted by arithmetic in the handler; shmembench's passes are the closed form of 32 consecutive words |

So the closed forms and the table of shapes already take the regular patterns; what is left is (a) warps on no stride that the table has not seen (random gathers, data-dependent windows), counted by general code that costs 280 to 630 cycles, (b) the table's own hit path (the 112 M and 139 M hits), and (c) global warps of a few runs of consecutive words.

### What changed (everything gives the same numbers as before, lane for lane)
1. **Bank passes of a warp on no stride and not in the table** (`sass_bank_passes_widened`, include/vgpu/exec/mem_counters.hpp; every shared-memory miss of the tier ends here). In order: the closed forms of a stride (the offsets are widened to 64 bits only when the first, second and last lane say a stride is possible; one lane is one pass), then
   * `simd::bank_window_kernel` (AVX2, src/simd/simd.cpp): every word the warp touches within 128 consecutive words (rows of 32 words in registers: a lane's words are bits of the row they are in, the rows are OR-ed over the lanes with a compare each, and the most distinct words in one bank is the bit-sliced maximum of the four rows' columns: two rows share a bit, three, all four). 1 to 16 bytes, any mask, idle lanes ignored;
   * `detail::bank_passes_max_lanes` (1 to 4 bytes, no lane running into a second word): a bank holds at most as many words as it has lanes, so the passes are at most M = the most lanes any bank has, and they are M if the bank that has M lanes has M different words: one byte counter per bank, the lanes of that bank found by a compare of the banks as bytes, their words compared pairwise (M up to 8; otherwise, or when every such bank has two lanes on one word, the next form counts);
   * `detail::bank_passes_slots` (1 to 16 bytes): each bank keeps its first four distinct words in a row of a 1 KB array, a word is new unless one vector compare of the row finds it; a bank that needs a fifth word gives the warp up;
   * and last the table of words as before.
2. **The tier's shared-memory handler asks the table of shapes with an AVX2 form of `sass_bank_passes_offsets`** (`simd::bank_offsets_kernel`, called from `count_shared_offsets` when VGPU_FASTCOUNT is not off): the warp's key (the offsets less the first running lane's word) lives in four registers, the entry the instruction last used (`hint`) is compared with four loads and four XORs, only a warp that is not in that entry spills its key for the hash. The same table, the same entries, the same window counters, state for state (`test_bank_kernels` compares the two tables byte for byte).
3. **Sectors of lanes whose sectors rise** (`simd::sector_up_kernel`, before `sectors_monotone<true>` in `sectors_fast`): a run of consecutive lanes (several runs of consecutive words, a rising gather) is counted four lanes to a step: the sum over the lanes of (last sector - first sector) + (first sector is not the last sector of the lane before), the lane before taken from the same group rotated by one lane; a lane whose first sector is below the last of the one before gives the warp up (scalar forms count it); 1 to 32 bytes, addresses under 2^62.

Not changed: the global handler (`h_global`), the memory manager, the PTX interpreter's counting (it has no tier and no table of shapes in these programs), the counters' formulas. The new kernels are the end of `src/simd/simd.cpp`, a hot unit (`VGPU_HOT_UNIT_FLAGS`), after every other kernel, so none of the others moved inside the unit; `lint_hot_unit_layout` passes.

### Results

**Whole programs** (server2, `taskset -c 24`, `VGPU_THREADS=1`, `VGPU_GPU=nvidia/rtx3060`, SASS engine, tier on, JIT off; four builds run in turn in rotating order, 15 rounds (10 for shmembench and SYR2K, 5 for alignedTypes and FDTD-2D, whose rows are flat), each run alone under the builder lock, median wall; spread = (max - min) / median; p = Mann-Whitney, two-sided; load average under 10 for every run, but a CI runner and other sessions' builds shared the box and the spreads of 15 to 24 % are outliers from that, so medians are what is read). A = origin/main 735c2626, **A2 = the same build copied at the same time (the A/A control)**, F = this branch, N = F with every new path switched off (kernels registered as null, `widened` as before: the control for a flat or negative row, built from the same sources at the same time).

| program | runs A/A2/F/N | A s (spread) | A2 s (spread) | F s (spread) | N s (spread) | **A/F** (p) | A/A2 (p) | N/F (p) |
|---|---:|---|---|---|---|---:|---:|---:|
| histogram | 15/15/15/15 | 11.08 (24%) | 10.97 (24%) | 9.53 (19%) | 11.52 (20%) | **1.163 (0.000)** | 1.010 (0.633) | 1.209 (0.000) |
| radixSortThrust | 15/15/15/- | 31.15 (9%) | 31.36 (9%) | 26.07 (11%) | - | **1.195 (0.000)** | 0.993 (0.013) | - |
| sortingNetworks | 15/15/15/15 | 10.35 (22%) | 10.28 (22%) | 9.04 (16%) | 10.78 (3%) | **1.145 (0.000)** | 1.007 (0.120) | 1.192 (0.000) |
| mergeSort | 15/15/15/15 | 6.96 (13%) | 6.83 (12%) | 5.78 (12%) | 7.06 (10%) | **1.204 (0.000)** | 1.019 (0.003) | 1.221 (0.000) |
| convolutionSeparable | 15/15/15/15 | 8.38 (4%) | 8.37 (22%) | 8.07 (9%) | 8.64 (4%) | **1.038 (0.000)** | 1.001 (0.633) | 1.071 (0.000) |
| alignedTypes | 10/5/10/5 | 26.08 (18%) | 26.01 (12%) | 25.73 (17%) | 25.52 (17%) | **1.013 (0.345)** | 1.002 (0.903) | 0.992 (0.327) |
| FDTD-2D | 5/5/5/- | 71.95 (14%) | 73.64 (15%) | 71.75 (4%) | - | **1.003 (0.917)** | 0.977 (0.251) | - |
| SYR2K | 10/10/10/10 | 7.97 (4%) | 7.97 (22%) | 8.26 (15%) | 7.86 (7%) | **0.965 (0.000)** | 1.001 (0.212) | 0.951 (0.002) |
| shmembench | 10/10/10/10 | 3.55 (2%) | 3.53 (4%) | 3.60 (11%) | 3.61 (2%) | **0.986 (0.003)** | 1.006 (0.009) | 1.003 (0.970) |

Reading it: the four programs whose counting was the general code (histogram, radixSortThrust, sortingNetworks, mergeSort) run 1.14x to 1.20x faster (the control N is 1.19x to 1.22x slower than F, so the gain is not a layout draw), convolutionSeparable (hits of the table and the two-row sectors) 1.04x. A/A2 is 1.00x to 1.02x (mergeSort 1.019, p 0.003; radixSortThrust 0.993, p 0.013: the A/A noise floor of this box is about 2 %).

**The flat rows are not flat in A/F, and the control says why.** SYR2K: A/F 0.965 (p 0.000) with A/A2 1.001, and the control N is as fast as A (A/N 1.015) and 5 % faster than F: SYR2K counts every access by arithmetic in the handler (`count_global_progression` 134 M times, `count_run` 34 M; none of the functions of this round is called: instrumented run) so the same instructions run, and the 3.5 % is where the tier handlers sit in the unit (`h_global<false>` at 0x380d80 in A, 0x380640 in F, 0x3811c0 in N). shmembench: A/F 0.986 and N/F 1.003 (its passes are a closed form in `h_shared`): the same placement. alignedTypes (10 rounds A and F, 5 of A2 and N) 1.013 with A/A2 1.002 and N/F 0.992, FDTD-2D (5 rounds) 1.003 with A/A2 0.977: inside the A/A noise (about 2 to 3 % on these runs of 26 s and 72 s).

**Instructions and cycles** (`perf stat -e cycles:u,instructions:u`, server3, cpu 6, two runs each, interleaved; instruction counts repeat to the instruction):

| program | cycles A -> F (G) | instructions A -> F (G) |
|---|---|---|
| histogram | 65.24 -> 57.54 (0.882x) | 283.40 -> 255.83 (0.903x) |
| radixSortThrust | 185.26 -> 158.14 (0.854x) | 744.79 -> 652.76 (0.876x) |
| sortingNetworks (first version of this branch, the bank kernels without the AVX2 offsets function: three runs each) | 64.09 -> 62.74 (0.979x) | 261.80 -> 257.93 (0.985x) |

The other programs' `perf stat` runs waited behind another session's long runs for the lock on server3 for over an hour and were not taken; instruction counts of A, F and N at equal work are therefore only shown for these three.

**Per function** (`perf record -e cycles:u`, server3, one thread, shares of the program's own samples; the totals are not comparable between the two columns, A was recorded under a load average of 8 to 16 and F under 2 to 3, SMT neighbours inflate cycles; the shares are):

| program | counting functions, share of samples A | F | samples A | F | functions (A -> F, share of the program) |
|---|---:|---:|---:|---:|---|

**Per call** (microbenchmark, `g++ -O2`, one CPU of server2, 400 repetitions of 512 warps of the shape, minimum of five; ns per call; every column includes the conversion of the 64-bit addresses to the tier's 32-bit offsets, about 5 ns, which a real call does not pay):

| shape | `sass_bank_passes_offsets` A | F (all new paths) | F (AVX2 offsets function, cold table) |
|---|---:|---:|---:|
| 4-byte words, random over 64 KB | 95.5 | 48.9 | 48.4 |
| 4-byte words, random over 1 KB (histogram's) | 73.1 | 70.7 | 70.6 |
| 4-byte words, random over 128 words (a merge window) | 74.3 | 26.0 | 25.8 |
| half the lanes, random words | 63.0 | 42.0 | 41.7 |
| bitonic partners (table hit) | 12.9 | 13.0 | 15.0 |
| pairs every fourth word, two rows of 16 (table hit) | 7.9 | 7.7 | 8.1 |
| unit stride, broadcast (closed forms) | 8.0 | 8.0 | 8.2 |

| shape (global, 32 lanes of 4 bytes) | `distinct_sectors_uncapped` A | F |
|---|---:|---:|
| two runs of 16 words, 4 KB apart (convolutionSeparable's are 12 KB apart) | 24.7 | 14.9 |
| random words over 1 MB / 4 KB | 59.4 / 59.7 | 60.9 / 58.6 |
| unit stride, stride 12 bytes, stride 256 bytes, broadcast | 6.2 | 6.5 |

The kernels do not help the random-over-1 KB shape (a microbenchmark with the table warm and the code in cache): in the programs it is 8 to 9 % of histogram's time, because the code it replaced (a 64-row bitmap with a memory dependency chain through its rows) was slower in the programs than here.

**What it is worth on the 15-program pool of the class study** (354.7 s of user cycles; the nine programs of this round at their measured A/F, the other six, which touch none of these paths, at 1.00): **1.037x** (354.6 s -> about 342 s; 1.035x counting only the five programs this round changed). The five save 12.1 s (radixSortThrust 6.5 s, histogram 1.9 s, mergeSort 1.5 s, sortingNetworks 1.8 s, convolutionSeparable 0.4 s); the flat rows' placement noise is +-3 % and FDTD-2D (28 % of the pool) moved +0.3 %. This is the pool's memory-counting share (the class study: global 22.5 % + shared 10.6 %, counting 8 % of that) cut where it was general code; the closed forms and the table of shapes already took the rest. In instructions, which do not depend on placement: histogram -9.7 %, radixSortThrust -12.4 %, cycles -11.8 % and -14.6 % (perf stat).

### Tests (CMake-wired, `vgpu_unit_test`, framework-free) and mutants
* `test_bank_kernels` (new; also registered as `test_bank_kernels_simd_base` with `VGPU_SIMD=base`): each path against a definition written from scratch (the words the bytes of every active lane cover, the most distinct words in one of 32 banks), through `sass_bank_passes_widened` and `sass_bank_passes_offsets` (behind a table in four states: fresh, kept, the warp asked again) and each kernel directly. A kernel must take exactly the warps it is made for and decline the rest with `out` untouched (so a kernel that quietly switched off for a shape fails too): the window kernel takes every warp whose words lie within 128 consecutive words (spread 127 yes, 128 no, from every lowest word's place in its row), `max_lanes` every warp whose fullest bank has that many different words, `slots` every warp whose banks hold at most four words each. Families: 1.5 M random warps in windows of 1 to 4000 words at widths 1, 2, 3, 4, 5, 6, 8, 12 and 16 bytes, alignments and bases; 1 M random scatters and lanes on a pool of 2 to 13 words (some in one bank); the edges (spread 0 to 200 words from every lowest word 0 to 69; k words of one bank in k rows, k up to 7; four rows sharing every bit); every mask of 16 lanes (the other 16 full, empty, alternate or one) of 14 structured shapes (unit, strides 2 and 33, bitonic partners, pairs, two rows of 16, groups on one word, bytes in words, 8 and 16-byte pairs and vectors, unaligned 4 and 8, four rows in one bank); idle lanes holding zero, noise, all ones or the first lane's word; offsets within 40 of 2^32 and at 0xFFFFFFF0 and after; widths 17 to 64 and wide spreads (they go to the table); strides -330 to 650 bytes at five widths; and a 600,000-call sequence on two tables, one through `sass_bank_passes_offsets` and one through the AVX2 kernel, equal after every call (answer, `skip32`, `last32`, `lookups32`, `hits32`, the hints) and every entry of both tables byte for byte every 257 calls (shapes asked again and again, evictions, random warps, the table going quiet and waking, partial masks, widths 1 to 70, offsets with bit 31, 30 or both set, a hint that is null, right, wrong or past the table).
* `test_sector_kernels` (new; also under `VGPU_SIMD=base`): `sector_up_kernel` against a set of sectors, and `distinct_sectors` / `distinct_sectors_uncapped` through it: 600,000 warps of two to four segments of consecutive words (segments far apart, adjacent, overlapping, sharing a sector, or going back), 500,000 strided and jittered gathers, 300,000 warps with one pair swapped or one lane moved back by a byte, a sector or many, first lane in sector 0 and at the top of what the kernel takes, every run `[lo, lo + n)` of 32 and 64 lanes at four widths, every mask of 12 lanes, idle lanes holding anything, widths past 32 and addresses at 2^62 and the top of the space.
* Mutants (each built with `g++ -O2` from the unit and the test, `src/simd/simd.cpp` linked in, pristine passes): 28 of the window, max-lanes and slots paths (an off-by-one in the last byte of a lane, a word short in the loop, idle lanes read, the four-row or three-row case dropped, no wrap guard, the spread bound at 126 and at 128, a wrong row compare, a wrong pair of rows, bank bits masked to 4, a clamp missing, a bound of 9 lanes, the pairwise check skipping a pair, the fifth word accepted, the valid mask one slot wider, the maximum missing a step, ...) and 11 of the sector kernel (carry 0 for 'no lane before', no backward check, tail lanes counted, the span in place of span - 1, no 2^62 guard, a half-lane blend, the sum without the tail mask, the fresh-sector test inverted, the compare reversed, span 33 taken, a gap in the run allowed) are all killed (the tests that still pass are listed in the log: 0 to 8 of 9). 11 of the AVX2 offsets kernel (the compare skipping its last vector, `last32` not updated, hits counted on every call, 65-byte accesses taken, the partial-mask blend inverted, the skip counter not decremented, bit 30 treated as bit 31, the hash reading another lane, the entry's width not stored, the hit counter not reset, `idx != first` dropped) and 3 of `widened`'s stride filter: all killed except the three that change no count: `idx != first` dropped (the entry compared again is the entry that just failed), the stride filter's last-lane test one lane early (a weaker necessary condition: warps that are not on a stride then widen and are turned away by the closed form), and the filter never true (the strided warps are counted by the kernels instead).
* Also run, passing (Release, server2): `test_memfast`, `test_memcount_forms` (also `VGPU_SIMD=base`), `test_bank_memo`, `test_bank_bitmap`, `test_sass_access_count` and the 57 tests of `ctest -R 'test_(memory|ptx_mem_runs|ptx_paths|sass_|memfast|memcpy_chk|memalloc|memrmw|memhot|gather|load_batch|warp_stride|memcopy|locallayout|memcount|bank|sector|atom_rmw|simd|unsupported|tier)|lint_hot'` (the SASS tests, `test_tier`, `test_simd`, the memory manager's, `lint_hot_unit_layout`). `test_bank_kernels` and `test_sector_kernels` under ASan+UBSan (`-O1`, halt on error, leaks on, both SIMD levels) and `test_bank_kernels` under TSan (`setarch -R`; single-threaded code, the table is the host thread's) on server3: pass. Compiled with g++-12 `-Wall -Wextra` (simd.cpp, exec.cpp, both tests): no warnings. The e2e programs that need `nvcc` are left to CI.

### Not done, measured and not kept, caveats
* **The table's hit path is still the biggest counting cost of sortingNetworks and convolutionSeparable** (`bank_offsets_k` 10.1 % and 7.7 % of the cycles, plus `count_shared_offsets` 5.3 % and 3.4 %). The AVX2 form took it from `sass_bank_passes_offsets` 12.9 % / 9.6 % but `count_shared_offsets` is no longer inlined into `h_shared` (a call and the thread's table pointer, `__tls_get_addr` through the library's dynamic TLS model): `[[gnu::always_inline]]` on it was tried in round sp38 and not kept (inside the noise of the control build; see "Round sp38 (follow-ups)"). An initial-exec TLS variable would remove the call into the dynamic loader but a library that `dlopen`s may not get static TLS: not tried.
* **Random scatters are still 60 to 70 ns a call** (histogram: `sass_bank_passes_widened` 15 % of its cycles after, from 20 %): the most-lanes form does one byte counter per lane and a pairwise word check, and it gives up when the fullest bank has two lanes on one word (then four-word rows, then the table). A sort of the 32 (bank, word) keys in AVX2 registers would be an estimated 80 cycles for any spread (about 300 vector operations by a count on paper; not written); the row-compare form of the window kernel costs 12 vector operations a row a lane group, so it stops paying past ~8 rows (the kernel takes 4).
* `bank_passes_slots` was no faster than the table in the microbenchmark (56 ns against 52 ns for random words) and is faster in the programs only because it needs 1 KB of stack rather than 16 KB (radixSortThrust before this round: the table 633 cycles a call, 17 % of the program); it is now the third fallback (inlined into `widened`).
* Sectors of random gathers (`sectors_hash`, `sectors_window`) and of a thread-private array are unchanged (59 ns for a warp of random words); the 22.5 M and 25.8 M warps of two or three runs that were `sectors_monotone`'s 120 to 170 cycles are the kernel's now (convolutionSeparable `sectors_fast` 4.5 % to 1.8 % + `sector_up_k` 1.8 %).
* The PTX interpreter (VGPU_SASS=0) has no tier and its table of shapes is the 64-bit one: not touched.
* Caveats. (1) **Layout lottery**: the new code sits in two hot units (`exec.cpp`: the call to the kernel; `simd.cpp`: the kernels, at the end after every other kernel so none of them moved inside the unit; and `mem_counters.hpp`'s inline functions in every unit that includes it). `exec.cpp`'s tier handlers moved (`h_global<false>` is at 0x380d80 in A, 0x380640 in F and 0x3811c0 in N; `step_warp` at 0x3ab400, 0x3ab0c0 and 0x3ab7c0), and the control N is itself 3 to 4 % slower than A on sortingNetworks and histogram while running the same instructions: read A/F for what a merge would give, N/F for what the code did, and expect +-3 % on any program in either from placement alone. (2) Timings were taken on a shared box (one builder lock shared with two other sessions, load average 4 to 10); medians of 15 interleaved rounds, spreads in the table. (3) radixSortThrust seeds its input with `clock()` and its onesweep kernels count the polls of a decoupled look-back at 4 threads, so its counters are only comparable at 1 thread with `clock` fixed (an `LD_PRELOAD` of a `clock()` returning a constant; A against A has the same differences at 4 threads, in the same seven fields, which are the polling loop's instructions and loads and nothing else). (4) The class study's pool is 28 % FDTD-2D, 14 % alignedTypes: both are untouched by this round (their counting is `count_run` / `count_global_progression` in the handler, 2 to 3 % of their cycles), which is why the corpus-wide figure is small.

## Round sp37 (dispatch loop and block setup): measured per line, one hot line removed, the rest measured and not kept (branch `sp37-dispatch`, base `origin/speedups-combined` 48d3e8df)

Question: pooled over 15 real NVIDIA programs (class study of 2026-11-06) dispatch, the run loop and scheduler around the handlers, is 20.7 % of the cycles (about 4.55 ns a warp instruction) and block setup 4.8 %. Cut both without changing a bit.

Same-bits rule: `VGPU_COUNTERS_FILE` (pid removed), `VGPU_KERNEL_DIGEST` and the exit status of every program timed equal those of the base at 1 and 4 threads (the standard output is compared with its timing lines removed; black-scholes, SYR2K, LU and FDTD-2D print a bare timing number on a line of its own that the filter of this round's script did not drop at first, so their output hash differed on the first pass and was equal once it did). Host: server3 (Ryzen 7 PRO 8840U, Zen 4 mobile; `nice -n 19`, `-j3`, every build, test and timed run under `flock /tmp/pantheonsim-builder.lock`, one run at a time, cpu 6, `VGPU_THREADS=1`). Builds are Release/LTO with `-g1` (no frame pointers: the code is the shipped code). A build is 8 minutes (libvgpu.a, then the LTO link of libcudart).

### Where the cycles of `step_warp` go, per source line (base build, `perf record -a -C 6 -e ibs_op/cnt_ctl=0/`)
Method. The first table of this round was a `cycles:u` profile at 1999 Hz, and it was wrong in the way the class study warned of: its hottest instruction (5 % of `step_warp`, the 16-byte `paddq` of the two instruction counters) was the skid of a long-latency instruction before it, and a build that removed what it seemed to show (below) changed nothing. The table here is AMD IBS op sampling (tag by cycles, 65,536 cycles a sample, precise instruction addresses, system wide on the pinned cpu, the process's samples kept), 14 programs, 4 to 7 M samples each, every sample mapped with `addr2line -i` to the outermost line that is in `tier_run_t` (inlined into `step_warp`) or in `step_warp`. Shares are of the samples that fall in `step_warp`; the last row is `step_warp`'s share of the program.

| where in step_warp (% of its samples) | FDTD-2D | LU | black-scholes | jacobi | mergeSort | interval | radixSort | sortingNet | pooled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pass setup (TCtx, regs_ok, glanes popcount, first/olim) | 10.7 | 11.7 | 10.9 | 14.3 | 9.3 | 8.0 | 13.6 | 7.9 | 10.8 |
| turn entry: runnable, slice, give_way, code_hint | 9.9 | 11.4 | 0.7 | 8.1 | 4.8 | 3.0 | 3.0 | 10.3 | 7.5 |
| turn entry: pick_group | 1.3 | 1.5 | 0.1 | 0.4 | 0.5 | 0.5 | 0.3 | 1.2 | 0.9 |
| stretch check and issue (k, steps, counters, tier_issue call) | 7.1 | 6.4 | 7.8 | 6.0 | 7.5 | 8.8 | 8.3 | 6.7 | 8.3 |
| store gate test | 5.0 | 4.4 | 2.2 | 2.1 | 3.7 | 3.1 | 2.0 | 5.7 | 3.9 |
| steps / limit (++w.steps > max) | 4.4 | 4.3 | 3.1 | 3.4 | 4.4 | 5.3 | 4.3 | 5.9 | 4.5 |
| one-op path: guard, lane count | 5.5 | 5.2 | 5.0 | 3.5 | 6.2 | 12.9 | 9.2 | 16.5 | 6.7 |
| one-op path: counters (3 words) | 4.0 | 3.6 | 3.0 | 2.1 | 4.2 | 4.7 | 4.7 | 8.0 | 4.5 |
| one-op path: indirect call, tier_stats test, ++o | 4.8 | 4.1 | 3.3 | 2.5 | 5.4 | 6.0 | 5.4 | 9.0 | 5.2 |
| cont ops and BRA taken by the whole group (in-loop continue) | 14.4 | 15.3 | 11.7 | 12.8 | 13.9 | 15.8 | 8.8 | 9.2 | 14.3 |
| run end: BRA, BSSY, EXIT, BAR, BSYNC, YIELD (control) | 25.7 | 24.9 | 28.9 | 27.1 | 26.6 | 20.7 | 24.5 | 15.4 | 23.3 |
| chain: flag tests, runnable, next group | 5.4 | 5.4 | 19.4 | 15.4 | 10.8 | 9.4 | 14.2 | 2.4 | 8.2 |
| chain: pick_group | 0.8 | 0.6 | 3.7 | 2.0 | 1.9 | 1.5 | 1.5 | 0.3 | 1.1 |
| step_warp, % of the program's cycles | 24.0 | 25.5 | 20.2 | 25.6 | 20.8 | 15.9 | 16.1 | 16.9 | 18.1 |



How to read it: the run loop's cost is not in the one-op path the brief expected (issue loop 47 % in the class study's line split; here the stretch check, store-gate test, step count, guard, counters, call and `++o` together are 33 % of `step_warp`, 4.5 to 8 % of a program) but in what happens once per call of the chain, and a call is short: mergeSort runs 354 M warp instructions in 94.9 M calls (3.7 instructions a call; `VGPU_TIER_STATS=2`: 86 M of them started by the chain). Pass setup, the in-loop continue past S2R/SEL/BRA, the run's end (BRA, BSSY, BSYNC, EXIT, BAR: 23 %, the largest line group) and the chain's tail are 56 % of `step_warp`, about 10 % of the corpus. `pick_group` is 2 % of `step_warp`, not the 21 % of the first (skid-based) table; the machine runs this loop at 3.95 instructions a cycle (mergeSort: 136.7 G instructions in 34.4 G cycles, 69 M branch misses = 4 % of the cycles), so what a change has to take out is instructions, and a change that removes a stall rather than instructions shows nothing (below).

One line stands out in FDTD-2D and LU (the two programs that are 36 % of the pool): the loop that moves the lanes that stay when some lanes of a warp leave at an EXIT (`for (Mask m = group & ~go; ...) w.pc[...] = here + 16;`, base `tier_ops.inc:2727`) is 14.5 % of FDTD-2D's `step_warp` samples (3.5 % of its cycles) and 8 % of LU's. Both have a guarded EXIT that lets some lanes of a warp go and keeps the rest: a loop of up to 31 trips with a data-dependent length.

### Changed
* **The lanes that stay after a partial EXIT** are given their next address by `tier_set_pcs` (one masked store, `set_pcs_partial`'s AVX2 blend where the host has it) instead of the lane loop. Same addresses for the same lanes: `src/sass/tier_ops.inc`, 5 lines, in the run loop, whose layout is therefore the only other thing that moved (`exec.cpp` is a hot unit; the object passes `lint_hot_unit_layout`'s convention, which `ctest` runs).
* **`tests/unit/test_sass_dispatch_counts.cpp`** (new, CMake-wired as `test_sass_dispatch_counts`, `_tier0`, `_simd_base` and `_jit`), written to hold the loop's counts and stops, op by op, to the interpreter's (`VGPU_FASTPATH=0`) whatever is done to the loop: see Tests.

### Results (15 interleaved rounds of three builds per program, medians of whole-process wall seconds, A/A in the same rounds)
A = `origin/speedups-combined` 48d3e8df, A2 = a second copy of A's library made at the same time, F = A plus this round's change (`68b0924c`). `VGPU_THREADS=1`, cpu 6, one process at a time under the lock, the order of the three alternating every round (ABC, CBA, ...); spread = (max - min) / median; p = Mann-Whitney, two sided.

| program | n | base s (spread) | base2 s (spread) | v5 s (spread) | A/A2 | A/F | p(A,F) |
|---|---:|---:|---:|---:|---:|---:|---:|
| FDTD-2D | 15 | 98.23 (1%) | 97.97 (6%) | 94.20 (3%) | 1.003 | **1.043** | 0.000 |
| LU | 15 | 29.37 (8%) | 29.33 (15%) | 28.85 (6%) | 1.001 | **1.018** | 0.000 |
| mergeSort | 15 | 7.42 (4%) | 7.37 (3%) | 7.31 (2%) | 1.007 | **1.014** | 0.000 |
| black-scholes | 15 | 20.22 (169%) | 20.22 (1%) | 19.94 (1%) | 1.000 | **1.014** | 0.000 |
| interval | 15 | 15.48 (4%) | 15.58 (11%) | 15.54 (2%) | 0.993 | **0.996** | 0.165 |
| sortingNet | 15 | 12.04 (2%) | 12.00 (5%) | 12.01 (2%) | 1.003 | **1.002** | 0.141 |
| jacobi | 15 | 19.46 (2%) | 19.40 (3%) | 19.29 (2%) | 1.003 | **1.009** | 0.001 |

A/A2 is within 0.7 % on every row (0.993 for interval, whose A2 spread was 11 %), so differences below about 1 % are not read. FDTD-2D's 4.3 % (98.23 -> 94.20 s, 15 of 15 rounds) is what the profile said (3.5 % of its cycles in the loop); LU's 1.8 % is less than its 3.2 % of IBS samples and in line with `perf stat` (cycles -0.8 %, instructions -1.5 %: a store-bound loop that waited on its stores costs fewer cycles than instructions suggest).

**Control for the flat and the unexplained rows.** interval and sortingNetworks are flat (0.994 and 1.006, p 0.165 and 0.110) and mergeSort, black-scholes and jacobiCudaGraphs gain 0.9 to 1.4 % with few EXITs. Control N is F's source with `if (stay) for (lanes) ...` the loop of A in the same place (same code size and layout up to the loop's body), built and timed in the same rounds as A and F:

| program | n | base s (spread) | control N s (spread) | F s (spread) | A/N | A/F | p(A,F) |
|---|---:|---:|---:|---:|---:|---:|---:|
| interval | 15 | 15.47 (3%) | 15.45 (5%) | 15.57 (3%) | 1.002 | **0.994** | 0.165 |
| sortingNet | 15 | 12.03 (3%) | 12.00 (3%) | 11.96 (3%) | 1.003 | **1.006** | 0.110 |
| mergeSort | 15 | 7.38 (3%) | 7.36 (6%) | 7.31 (4%) | 1.003 | **1.010** | 0.000 |

N equals A on all three (A/N 1.001 to 1.003), so the layout of the changed source is not what gives or takes the seconds: F's 1.0 % on mergeSort (p < 0.001) is the change's, its flat rows are flat. interval and sortingNetworks have a handful of partial EXITs per launch and nothing for the change to take.

**Same bits** (counters, kernel digest, exit status and the program's output with timing lines removed, base against F, `ab.py --same`, 1 and 4 threads, 10 programs): equal for LU, FDTD-2D, mergeSort, interval, sortingNetworks, histogram, simpleHyperQ, SYR2K at both thread counts and for jacobiCudaGraphs at 1; black-scholes, which prints its timing as a bare number the filter missed in that batch, differs in the output hash only (counters, digest and exit status equal); jacobiCudaGraphs at 4 threads differs in `VGPU_KERNEL_DIGEST` between **two runs of the base itself** (`base` against `base2`: `c96e03a1` against `3177deca`; counters and output equal), a program whose multi-thread digest is not reproducible before this round. 

**What the corpus gains** (the class study's pool of 15 programs, 354.7 s of user cycles, the programs not timed here at 1.00; times from the table above): FDTD-2D 98.8 -> 94.7 s, LU 29.9 -> 29.4, black-scholes 20.1 -> 19.9, mergeSort 8.6 -> 8.5, jacobiCudaGraphs 19.0 -> 18.8, interval, sortingNetworks unchanged: **354.7 -> 349.5 s, 1.015x** (pool without FDTD-2D, 255.9 s: 1.004x). Against the Amdahl bounds of the brief (dispatch 20.7 % + block setup 4.8 %: 1.14x if both were halved) this round took 0.7 % of the 25.5 %: the part of dispatch that is a single removable cost.


### Measured and not kept (built, same bits, A/B on server3)
Four builds of the dispatch loop before the IBS table existed, each from the one before, were timed against the base with an A/A copy (`ab.py`, interleaved, 2 runs each only, since the work stopped when the profile showed the lever was elsewhere), and `perf stat` for mergeSort:
* **v1**: the two instruction counters written as two words (`tier_count`, an empty `asm` on one of them) instead of the compiler's one 16-byte load/add/store that follows the 8-byte stores of the run-end paths; **v2** adds the guard read from the op's own copy of the three guard fields (the Instr's cache line is not touched for a guarded op), the lane count of a guard's lanes and of the group kept in two locals between ops and between calls of the chain, `pick_group`'s first test as one xor/or pass over the 32 addresses (`all_lanes_at`, SSE2) and the run loop's switches (`tier_stats_`, `tier_chain_`, `max_steps_`) read once into locals; **v3** adds the class counts of a stretch added when the class changes (`tier_issue`); **v4** adds one branch for the six reasons the chain stops. All four give the base's counters, digests and output.
* Result: mergeSort `perf stat` (cycles:u, instructions:u, `ls_bad_status2.stli_other`): base 34.23 / 34.55 G, 136.70 G, 1.23 / 1.27 G; v2 34.19 G, 136.58 G, 1.23 G; v4 34.52 G, 137.41 G, 1.18 G. **No change in cycles and none worth reading in store-forward failures** (the counters' 16-byte load was real but not the cost: the profile's hot `paddq` was skid). Wall (2 runs, A/A within 1 %): LU 29.3 -> 30.5 (v2) and 31.1 (v4), black-scholes 20.2 -> 20.6, interval 15.6 -> 16.5 (v1), the others within 1 %: slower on the two longest programs, which at that sample size is the layout lottery or a real loss and in neither case a gain. They were taken out; nothing of them is in the branch except the test.
* **Where the failed store forwards are** (`perf record -e ls_bad_status2.stli_other:u`, mergeSort, LU, black-scholes, interval, 960 k samples of 20,011 events): `icmp_mask_k` 15.5 %, `step_warp` 13.5 %, `h_global<false>` 7.6 %, `imad_k` 4.8 %, `pick_group_avx2` 3.8 %, `imad_wide_avx2` 3.8 %, `tier_dyn64` 3.4 %, `h_mov` 3.0 %, `ffma_k` 2.9 %: mostly the SIMD kernels loading a 32-byte row that a handler stored in two 16-byte halves or a masked blend. 1.1 to 1.2 G events for 354 M warp instructions on mergeSort (3.3 an instruction), 11.1 G on LU; not what this round could remove, and a lead for the kernels' round (the stores of `tier_store`/`tier_out` against the loads of the next op's kernel).
* **Block setup** (`init_block` 1.9 % and the unsymbolised leaf 2.7 % of the pool): the leaf is libc's `rep stosb` memset (a `rep stos` at libc offset 0x1a1efa of the server's libc: glibc's memset takes that path from 2 KB up) of the register file (`w.r.assign(nregs * 32, 0)`: 3.5 to 5 KB a warp) and of shared memory, 8.5 % of LU and 6.6 % of FDTD-2D (the `[unknown]` leaf of the class study's profile). A microbenchmark (`g++ -O2`, L1-hot buffers, minimum of 5 x 200,000 calls) puts libc's memset at 126 to 150 bytes a ns from 512 B to 48 KB, an AVX2 store loop at 126 to 153 and an SSE2 one at 80: the clear is at the speed of the store port, **no faster way to write the zeros exists, and the only lever is writing fewer**. That needs the rows a block cannot have written: the file is `k_.regs + 8` rows (the 8 are slack: 40 % of a 12-register kernel's file) and every row below `k_.regs` may be written. A bound from `MOp::maxreg` is sound for the ops the tier has a handler for, but not for the interpreter's (P2R, REDUX, the device-runtime calls and the wide tensor and tcgen05 forms write a number of rows that no field of the decoded op gives), and a dirty-extent tracked at the writes would add a store to every register write: not built. Verifying instead of assuming (scan the 8 slack rows for non-zero words and clear only when one is) saves a write of 1 KB for a read of it, about 13 % of the clear at best. The rest of `init_block` (the per-warp `w.reset()` and the 32 stores of `w.pc`) is 17 % of its self time on LU. Not changed.
* **Not tried**: BSSY and S2R-like ops as in-loop continues (a BSSY ends the call today: 10.8 M of mergeSort's 94.9 M calls); making SHFL and VOTE `cont` ops (jacobiCudaGraphs: `step_warp` is 30 % of its cycles, one SHFL ends every few instructions). Tier 2 uses `MOp::cont` in its generated code and in its cache key, so the set is shared with it and the change belongs with a JIT round.

### Tests (CMake-wired, `vgpu_unit_test`, framework-free) and mutants
* `test_sass_dispatch_counts` (new; also `_tier0`, `_simd_base`, `_jit`): hand-assembled sm_86 kernels, each run with the tier and with `VGPU_FASTPATH=0`, words, every counter before `inst_by_opcode` (instruction, thread-instruction and per-class counts included) and, for a launch that fails, the error message and code compared. Six tests: guarded runs (60 random kernels of 3 to 11 predicate groups, `@P0`/`@!P0`/none, a compare redefined mid-group, bounds that leave every lane or none; block sizes 32, 40, 20, 64), loops whose lanes leave at different trips (the group the chain picks changes from call to call; asserts the branch splits), `@P0 EXIT` and `@!P0 EXIT` for 0 to 99 lanes in the middle of a run (the lanes that stay move on: the change of this round), a split that leaves the high lanes (down to lane 31 alone) at another address, an op that faults after 0 to 300 ops (same error text), and **the step budget ended at every offset**: four kernels (guarded, loop, exit, a 300-op straight stretch longer than the loop's 256) with a budget of N for every N from 1 to the instruction count + 3, which must throw `ExecLimit` for N below the count and not from it, in the tier and the interpreter alike, and at 96 threads for the three offsets around it.
* `test_sass_alu_runs`, `test_sass_chain_cont`, `test_sass_store_gate`, `test_sass_lanes` (9), `test_tier` (9) pass on the build under test.
* Mutants (each compiled from a copy of the tree with one line changed, `exec.cpp` and the test linked against the pristine `libvgpu.a`, run on server3; 7 of them): the lanes that stay not advanced (`here` for `here + 16`), the wrong lanes advanced (`group & go`), none advanced (all three: `test_sass_dispatch_counts` fails the exit test and the budget test, which runs the exit kernel), the one-op path's step check off by one (budget test fails), a guarded op's lane count one too many (5 of 6 tests of `test_sass_dispatch_counts` and 2 of `test_sass_alu_runs` fail), the group's lane count fixed at 32 (5 of 6 and 5 of the alu tests fail), and the stretch's budget check `<` for `<=`: **survives, and is an equal program**: a stretch that ends exactly at the budget goes to the one-op path, which ends where the interpreter does. While the tests were written they were also run against mutants of the abandoned dispatch builds (below): the stretch's instruction counter one short and the group's lane count kept when the group changes (both killed by `test_sass_dispatch_counts`, and the second by `test_sass_chain_cont`), and `pick_group`'s new first test ignoring lanes 24 to 31 (killed by its own `test_sass_lanes` test only: `test_sass_dispatch_counts` and `test_sass_chain_cont` let it survive, which is why `high_kernel` was added; the second run of that mutant against it was not made).
* `lint_hot_unit_layout` (hot units on 64-byte function boundaries) passes on the build under test; `ci/jobs.sh`'s check passes.

### Caveats and what is left
* The headline is small and honest: **one line of the run loop, 1.5 % of the pooled corpus** (4.3 % of FDTD-2D, 1.8 % of LU). The pooled dispatch cost (20.7 %) is spread over about forty lines, none above 8 % of `step_warp` once the profile is exact; halving dispatch would need the per-call work (set-up, in-loop continues, run end, chain tail: 56 % of `step_warp`) cut by a design change, not by trimming: the calls are 3.7 to 8 instructions long, so each costs about as much as the ops it runs.
* **The first profile of this round was wrong** and cost four builds. `cycles:u` samples land on the instruction after the one that waits; the counters' `paddq` and `pick_group` looked like 5 % and 21 % of `step_warp` and were 0 % and 2 %. The lines in the class study's split table (47 % issue loop, 21 % group selection) are skid-based in the same way and should not be used to choose a change; the IBS table above is exact to the instruction. IBS is system wide (`perf record -a -C 6`), needs `perf_event_paranoid` <= 0 on the host, and was taken with `cnt_ctl=0` (cycles); a `cnt_ctl=1` (ops) profile of the same programs was taken too (not tabulated: the pooled shares of its line groups are within 4 points of the cycle ones).
* The EXIT line is a large cost in two of the 14 profiled programs (FDTD-2D, LU); the 0.9 to 1.4 % of mergeSort, black-scholes and jacobiCudaGraphs is not explained by their EXIT counts (mergeSort: 2.7 M of 354 M instructions). The control (N, above) rules out layout for mergeSort, the only one of the three it was run on; black-scholes and jacobiCudaGraphs have no control row and are read as small. The six programs of the first table ran in one batch and FDTD-2D in another, concurrently with it on the same lock, so the batches' absolute seconds are comparable only inside a batch.
* **Hosts and noise**: server3 is a shared machine (other sessions' jobs ran between ours: the lock only orders our own jobs); black-scholes' base median spread of 169 % is one outlier run of 54 s (a neighbour's job) in the 15; medians are unaffected. `perf stat` on LU: cycles 134.28 G -> 133.17 G (-0.8 %), instructions 515.49 G -> 507.59 G (-1.5 %).
* Not run: the AMD engine (the change is in the SASS tier only), the PTX engine, tier 2 timing (the `_jit` variant of the new test passes only where the JIT can compile, as the other tests of that list), threads above 4.
## Round sp38 (follow-ups): host writes in the caching budget, the cheap JIT form as the default, `always_inline` for the shared-memory table (branch `sp38-followups`, base `speedups-combined` 48d3e8df)

Three follow-ups of the pieces of the combined speed-up PR (#544), in the order they were asked for. Host: server2 (Zen 5, Ryzen 9 9950X, `nice -n 19`, `-j3`, Release with LTO, GCC 13; every build and every timed run under `flock /tmp/pantheonsim-builder.lock`, one job at a time; production Postgres and two CI runners shared the box, so a timed run waited whenever the load average was above 8-9). Programs of the corpora (PolyBench/GPU, Rodinia, HeCBench, CUDA Samples) as in round sp35; the e2e programs of the JIT's gate were compiled with nvcc 13.0 elsewhere (compile only) and run through `run_sass_jit.sh` with `VGPU_TIER_WORK`. Builds: **B0** = `origin/speedups-combined` untouched (the same-bits reference), **A** = this branch, **A2** = a byte-identical copy of A made at the same time (the A/A control), **F** = A with `always_inline`, **N** = A with a no-op line added in the same function (the layout control). Raw data and logs: `docs/perf-design/data/sp38/`.

### 1. Result caching: the hashing of host writes is part of the budget

**What was wrong.** Round sp35's one miss-path failure was `samples/simpleCudaGraphs` at +14 % cold: the cache's own accounting showed 28 ms of recording and 0.4 ms of fixed cost against 160 ms of added wall time, and the rest was the hashing of every host write (a `cudaMemcpy` of the program's buffers is an event of the history with the hash of its bytes, 14 GB/s, not under any budget).

**What changed** (`src/runtime/memo.cpp`, `include/vgpu/runtime/memo.hpp`):
* `DeviceMemo::on_write` asks `event_budget_allows(len)` before hashing: the estimate (`len` at the measured hash rate, plus 0.3 us) must fit `event_ns + estimate <= VGPU_MEMO_BUDGET x kernel time + VGPU_MEMO_EVENT_CREDIT_MS` (8 ms by default: about 110 MB hashed for free, which is what a program uploads before its first launch). A write that does not fit **is not hashed and poisons the history** (a fresh root no other run has), so the launches after it can never match a stored result; they still run, are recorded when the budget allows, and give the same answers as without the cache. What is lost is the hits after the write, never a result.
* What the hashing cost is measured (a write of 64 KiB or more is timed, smaller ones are charged by estimate, 0.3 us plus the bytes at the hash rate) and is part of the pool the record decision draws on (past its own credit, so uploads before the first launch do not starve recording), as the brief asked: `event_ms` and `events_skipped` are in the statistics line.
* A cache that went dormant (more than 256 launches, none hit, fixed cost above 5 % of kernel time) keeps no history at all now: `on_write` returns at once.
* `VGPU_MEMO_BUDGET=inf` still means everything is hashed and recorded (tests, the unbudgeted differential).

**Per-process cost of turning the cache on.** Measured again on the final build (`samples/vectorAdd`, 60 alternations of off, off, memory, memory + statistics file, cold directory, cold + statistics file, warm directory; medians 2.584, 2.587, 2.635, 2.657, 2.671, 2.690, 2.647 ms): **+0.05 ms in memory, +0.09 ms with `VGPU_MEMO_DIR` (+0.11 with a statistics file)**. The 0.16 ms / 0.32 ms of round sp35 does not reproduce on this build (the build id, the environment hash and the store are read lazily; a `strace` of a one-launch program shows 2 more syscalls than off without the directory: one `mkdir`, one `openat`), so nothing was changed here. A program that records a launch and writes its first entry (the fan-out directory, the temporary file, the rename) pays on top: the cache's own accounting says 0.06 to 0.08 ms for `samples/template`, and its wall is +1.1 ms on a 1.3 ms run (file-system work of the first entry, not compute). That, and the process start, is why the programs under 50 ms are read separately (table below).

**Miss path, corpus** (`memo_diff.py --mode overhead`, default budget, cold directory every time, `VGPU_THREADS=1`, pinned core, 5 alternating runs of off, off again (A/A), cold, warm; the 13 longest were not in this pass; 15 runs for the 35 programs under 50 ms; build A):

| | programs | off s | A/A | **cold** | warm |
|---|---:|---:|---:|---:|---:|
| 54 programs of round sp35's table (same programs) | 54 | 42.7 | -0.07 % | **+0.84 %** (round sp35: +1.09 %) | 0.81x |
| all 50 ms or more | 75 | 338.0 | +0.32 % | -11.8 % (programs that hit in their first run are faster: histogram, convolutionSeparable, LU, ...) | 0.77x |
| ... of which no hit in the cold run | 68 | 276.8 | +0.19 % | **+1.13 %** | |
| four of the 13 longest (3 runs) | 4 | 349 | | 2MM +0.1 %, FDTD-2D +4.2 %, GRAMSCHM +1.1 %, bf16TensorCoreGemm -0.5 % (A/A +0.1 %, +0.0 %, -0.1 %, +2.5 %) | |

Round sp35's five programs over 5 %: `simpleCudaGraphs` **1.140 -> 1.044** (its events: 28.6 ms hashed, 2 writes skipped, 22 of 24 launches not recorded), `streamOrderedAllocation` 1.068 -> 1.065 (8.1 ms of events on a 0.2 s program: the credit), `simpleMultiGPU` 1.062 -> 1.002, `sc_gpu` 1.056 -> 1.035, `asyncAPI` 1.053 -> 1.000. Over 5 % now: `streamOrderedAllocation` +6.5 %, `needle` +5.6 % (255 launches of 0.9 ms; one recorded), none over 10 %. The PolyBench programs that upload and never record (2DCONV, 3DCONV, ATAX, BICG, GEMVER, GESUMMV, MVT) were +0.7 to +4.2 % cold before and are -1.5 to +0.5 % now: their writes were hashed for nothing, and the 18 programs that had at least one write skipped are listed in the per-program table. The warm run of `simpleCudaGraphs` loses its 5 hits (the writes that carried them are the skipped ones); a few other programs record less than in round sp35 (`rodinia/lud` 30 -> 11 hits, `threadFenceReduction` 6 -> 1): launches are shorter since round sp36, and the budget is a share of kernel time.

**The 35 programs under 50 ms** (run is mostly process start; 15 alternating runs): median cold cost **0.27 ms** (0.16 ms for the 17 that record nothing, 0.29 ms for the 18 that record, worst 1.16 ms: `samples/ptxjit`, 1.7 ms off), total +5.8 % of 229 ms; 8 of 35 are over 10 %, every one of them a 1 to 4 ms run. The gate is read on programs of 50 ms or more, as in round sp35.

**Pantheon workloads** (A10, 1 GiB, 3 s, `--kernel_loops 2 --warmup_iters 1 --grid_size 4`; 39 workloads, build A): **24 of 39 workloads at 2x or more warm, 22 at 10x or more, 13 at 100x or more** (round sp35: 24, 22, 14: `march_test` 32,630x against 41,060x, `fused_attention` 262x against 289x, the same order of the headline), the 12 that cannot hit run at 0.949 to 1.000 (median 0.970) of the uncached launch rate cold. 532,629 hits across the 39 were run for real too (`VGPU_MEMO_VERIFY=2`): **0 mismatches**.

**Differential** (`memo_diff.py --mode correctness`: stdout and written files, exit code, kernel digest, counters, hash of all device memory at exit; off x3, cold, warm, warm verified):
* default budget, 111 programs (the 13 longest left out): **1 differs, `samples/radixSortThrust`, which differs between two runs without the cache** (clock-seeded input; the control of round sp33); warm hits 7,722 of 41,178 launches (18.8 %: the default budget records less than the unbudgeted run of round sp35, which hit 93.4 %); **8,643 hits verified against a real run, 0 mismatches**;
* the poisoning paths forced (`VGPU_MEMO_EVENT_CREDIT_MS=0`: every host write over its credit poisons), 58 programs that record launches: **0 differ**, 1,180 warm hits of 32,166 launches, 787 verified, 0 mismatches;
* unbudgeted (`VGPU_MEMO_BUDGET=inf`, every miss recorded and every host write hashed), the same 58 programs: warm hits 30,611 of 32,166 launches (95.2 %), **30,610 hits verified against a real run, 0 mismatches**; stdout, files, exit code, counters and final device memory equal to the uncached run in all 58; the kernel digest of `samples/simpleCallback` (host callbacks) differs between runs *without* the cache as well (three off runs of the stress pass give two values, with 0 hits in the run that differs), so it is not read as a cache difference (the table's program line says DIFFERS: digest; data `docs/perf-design/data/sp38/memo-corpus-correctness-unbudgeted.md`).

**Tests** (`tests/unit/test_memo.cpp`, `build/test_memo`: 49 tests, 44 before; all CI-wired through the existing `test_memo`):
* `host_writes_over_the_budget_poison_and_never_make_a_wrong_hit`: credit 0, a second of recording credit: three runs (bytes 1, bytes 5, bytes 1 again) give the answers of a device without the cache, never hit, every one has `events_skipped >= 1`, the reason `event-budget`, `event_ns` 0 and its launch recorded (it is the history that is poisoned).
* `host_writes_within_the_budget_are_hashed_and_a_second_run_hits`: the same with a second of credit: hashed (`event_ns > 0`), the same bytes hit, other bytes do not.
* `a_big_write_over_the_budget_poisons_and_the_small_writes_after_it_are_hashed_again`: 8 MiB against a 1 ms credit is skipped (exactly once), the 64-byte writes after it are hashed, nothing carries over from the first run (`disk_hits` 0), a loop of unchanged launches still hits in the run.
* `the_hashing_of_host_writes_stops_at_its_credit`: 700 writes of 1 MiB against 10 ms: some are skipped and the time charged stays under 30 ms.
* `a_dormant_cache_keeps_no_history_and_the_answers_are_the_same`: with `dormant_after = 4` and a fixed-cost share of 0 (test-only fields of `memo::Config`, so the quiet state is reached at once and not by timing) 8 launches that never hit make the cache dormant; a host write after that does not move the state, and every step of the program equals the same program without a cache.
* **Mutant** `event_budget_no_poison` (the write the budget refuses becomes an event without its bytes, which is the obvious wrong rule): `every_mutant_is_caught` runs a run with bytes 1 and then a run with bytes 5 against the same cache directory and requires the second to answer differently from the truth or to hit: **caught** (30 mutants, all caught; the second new one, `dormant_keeps_history`, is the dormant cache that still hashes the writes it is told about, caught by the dormancy test's scenario; `the_unmutated_cache_passes_the_same_scenarios` runs every scenario with no mutant: clean). `build/test_memo` and the e2e matrix are otherwise unchanged.

### 2. JIT: the cheap form of the generated code is the default

`VGPU_JIT_RESIDENCE` and `VGPU_JIT_SPECIALIZE` now default to **off** in `Config::from_env()` (`1` turns each on; `0` or unset leaves it off); the members of `jit::Config` keep the full form because the generator's tests (`test_jit_resid`) compare the cheap form against it. The key of a cached object includes both settings, so old cache entries are simply not found.

**Matrix, rerun for the new default** (build A, `VGPU_JIT=1`):
* `ctest`: `test_jit`, `test_jit_share`, `test_jit_resid`, `test_sass_{chain_cont,store_gate,alu_runs}_jit`, `test_tier`, `test_memo`: 8 of 8 pass.
* The seven `e2e_sass_jit` shards (tier, loops, steps, steps_gather, steps_shared, random, modes): **all pass** (`jit-e2e-shards.log`), with a new stage in `modes` (`form`, `run_sass_jit.sh`): the full form, residence alone and specialisation alone each print what the tier prints with generated regions running, so both forms stay CI-covered after the flip.
* Same bits, 22 programs x {1, 4} threads x {tier, JIT cold, JIT warm} against B0's tier-only run (`verify.py`; `jit-same-bits-default-form.log`): **132 comparisons, 120 equal**; the 12 that differ are the three programs where the reference differs from itself, the same ones as in round sp33: `radixSortThrust` (counters and digest, 1 and 4 threads), `jacobiCudaGraphs` (digest, 4 threads), `segmentationTreeThrust` (counters, 4 threads). **Control** (`jit-same-bits-controls.log`): B0's tier against B0's tier and against A's tier on those three programs at 1 and 4 threads differs for exactly the same programs and thread counts, with equal output. The full form (`VGPU_JIT_RESIDENCE=1 VGPU_JIT_SPECIALIZE=1`), 12 JIT-active programs x 2 thread counts x cold and warm: **48 of 48 equal** (`jit-same-bits-full-form.log`).
* Timing (`sp33.py`, warm cache, 5 interleaved runs of tier, tier again (A/A), JIT cheap, JIT full for the 16 programs under 30 s; 3 runs of tier, cheap, full for the six longer ones; one cold run each, compile on the launching thread; medians of whole-process wall seconds, spread = (max - min) / median in brackets):

| program | runs | tier s (spread) | JIT, cheap form (default) s | JIT, full form s | A/A (tier/tier) | **tier / cheap** | tier / full | full / cheap (time; above 1: cheap is faster) | cold cheap s (compile ms) | cold full s (compile ms) | compile cheap / full |
|---|---:|---|---|---|---:|---:|---:|---:|---|---|---:|
| polybench/SYRK | 5 | 4.13 (0%) | 3.76 (1%) | 3.71 (7%) | 1.003 | **1.100** | 1.113 | 0.988 | 5.16 (1, 1451) | 6.18 (1, 2599) | 0.56 |
| polybench/SYR2K | 5 | 7.80 (1%) | 7.09 (0%) | 6.97 (1%) | 1.003 | **1.099** | 1.118 | 0.983 | 7.88 (1, 952) | 8.20 (1, 1434) | 0.66 |
| rodinia/gaussian | 5 | 3.06 (1%) | 3.01 (2%) | 3.24 (1%) | 1.007 | **1.016** | 0.947 | 1.073 | 4.41 (2, 1461) | 5.16 (2, 2008) | 0.73 |
| samples/convolutionSeparable | 5 | 7.92 (2%) | 7.90 (1%) | 7.96 (2%) | 0.997 | **1.002** | 0.994 | 1.008 | 7.85 (0, 0) | 7.86 (0, 0) | - |
| samples/mergeSort | 5 | 5.76 (1%) | 5.22 (1%) | 5.25 (1%) | 1.009 | **1.105** | 1.098 | 1.006 | 8.76 (4, 3604) | 10.24 (4, 5092) | 0.71 |
| samples/sortingNetworks | 5 | 8.64 (0%) | 8.08 (0%) | 8.07 (1%) | 0.999 | **1.070** | 1.070 | 0.999 | 9.78 (3, 1685) | 10.26 (3, 2183) | 0.77 |
| samples/interval | 5 | 12.36 (11%) | 12.00 (15%) | 11.83 (17%) | 0.996 | **1.030** | 1.045 | 0.986 | 14.57 (3, 2956) | 16.48 (3, 4516) | 0.65 |
| samples/histogram | 5 | 9.41 (1%) | 8.26 (1%) | 8.17 (1%) | 1.006 | **1.139** | 1.152 | 0.989 | 10.91 (3, 2558) | 11.84 (3, 3626) | 0.71 |
| samples/jacobiCudaGraphs | 5 | 14.77 (0%) | 14.84 (1%) | 14.83 (1%) | 0.994 | **0.996** | 0.996 | 1.000 | 14.71 (0, 0) | 14.80 (0, 0) | - |
| samples/simpleHyperQ | 5 | 9.20 (1%) | 4.65 (6%) | 5.41 (2%) | 0.838 | **1.980** | 1.700 | 1.165 | 5.28 (1, 372) | 5.84 (1, 413) | 0.90 |
| samples/reduction | 5 | 6.71 (2%) | 5.32 (1%) | 5.21 (5%) | 1.029 | **1.262** | 1.289 | 0.979 | 6.18 (2, 909) | 6.47 (2, 1110) | 0.82 |
| samples/transpose | 5 | 6.61 (21%) | 5.67 (19%) | 5.48 (19%) | 1.045 | **1.165** | 1.207 | 0.966 | 9.00 (8, 3951) | 9.81 (8, 4830) | 0.82 |
| samples/cudaTensorCoreGemm | 5 | 7.58 (2%) | 7.58 (2%) | 7.59 (1%) | 0.982 | **1.000** | 0.999 | 1.002 | 7.50 (0, 0) | 7.49 (0, 0) | - |
| hecbench/black-scholes-cuda | 5 | 15.86 (1%) | 15.95 (1%) | 16.01 (0%) | 1.000 | **0.994** | 0.991 | 1.004 | 18.65 (1, 2672) | 20.14 (1, 4087) | 0.65 |
| hecbench/shmembench-cuda | 5 | 3.47 (1%) | 3.48 (1%) | 3.46 (0%) | 1.003 | **0.997** | 1.001 | 0.995 | 3.47 (0, 0) | 3.49 (0, 0) | - |
| hecbench/atomicAggregate-cuda | 5 | 2.98 (1%) | 2.83 (0%) | 2.80 (2%) | 0.999 | **1.055** | 1.063 | 0.992 | 3.36 (1, 521) | 3.47 (1, 652) | 0.80 |
| polybench/LU | 3 | 21.04 (0%) | 19.19 (0%) | 19.13 (0%) | - | **1.097** | 1.100 | 0.997 | 20.61 (2, 1332) | 20.89 (2, 1776) | 0.75 |
| samples/alignedTypes | 3 | 25.08 (1%) | 20.74 (1%) | 20.73 (1%) | - | **1.209** | 1.210 | 1.000 | 26.14 (8, 5479) | 28.15 (8, 7469) | 0.73 |
| samples/radixSortThrust | 3 | 25.10 (0%) | 25.42 (0%) | 25.40 (0%) | - | **0.988** | 0.988 | 0.999 | 27.90 (1, 2307) | 29.26 (1, 3780) | 0.61 |
| samples/segmentationTreeThrust | 3 | 12.40 (0%) | 12.14 (1%) | 12.28 (0%) | - | **1.021** | 1.010 | 1.011 | 42.26 (22, 29986) | 60.62 (22, 48322) | 0.62 |
| polybench/FDTD-2D | 3 | 67.00 (1%) | 61.19 (0%) | 60.86 (0%) | - | **1.095** | 1.101 | 0.995 | 62.59 (3, 1415) | 62.74 (3, 1687) | 0.84 |
| hecbench/fpdc-cuda | 3 | 6.96 (34%) | 5.39 (14%) | 6.27 (10%) | - | **1.291** | 1.110 | 1.163 | 5.69 (1, 1325) | 6.53 (1, 2048) | 0.65 |


geometric mean tier / cheap **1.1092**, tier / full 1.0957, full / cheap 1.0124; worst tier / cheap 0.988 (samples/radixSortThrust), best 1.980 (samples/simpleHyperQ); A/A 0.838 to 1.045 (geomean 0.9933 over 16 programs); compile ms summed over the programs that compile: cheap 64935, full 97632 (cheap / full 0.665); per program cheap / full 0.56 to 0.90; sums of medians: tier 283.8 s, cheap 259.7 s, full 260.7 s; cold wall over the tier's: cheap geomean 1.119, full geomean 1.213

The A/A reading: SYRK, SYR2K, gaussian, convolutionSeparable, mergeSort, sortingNetworks, histogram, jacobiCudaGraphs, cudaTensorCoreGemm, black-scholes, shmembench, atomicAggregate are 0.994 to 1.009; `simpleHyperQ` (0.838: its tier time is bimodal, 9.2 and 11.0 s), `transpose` (1.045, spread 21 %), `interval` (spread 11 to 17 %) and `fpdc-cuda` (spread 34 %) are outside the noise floor of this box and are not read. Over the other 18 programs the cheap form is **1.0665 over the tier against 1.0656 for the full form (full over cheap time: 1.0009)**: the cheap form loses nothing at run time, and its compile is **0.67x** of the full one over the programs that compile (0.56 to 0.90 per program; 65 s against 98 s in total), so a cold run is 1.119x of the tier against 1.213x (round sp33: 1.22x).
* **Verdict: defaults flipped.** The only programs where the cheap form takes more than 2 % longer than the full one are `reduction` (1.022, spread 4.7 %) and `transpose` (1.035, spread 19 to 21 %): noise.
* **Test and mutants** (`jit_the_product_default_is_the_cheap_form_and_the_knobs_turn_the_full_form_on` in `tests/unit/test_jit.cpp`, CI-wired as `test_jit`): it checks `from_env()` for unset, `1`, `0` and the empty string, and that a bare `jit::Config` is the full form. Mutant 1 (the product defaults back to the full form): `test_jit` fails; mutant 2 (the knobs never turn the full form on): `test_jit` fails; the pristine source passes (`build-mut`, three builds of `test_jit` under the lock: rc 1, 1, 0). The e2e `form` stage is the check that the full form still runs and still equals the tier; the generator's own mutants (`ctl_*`, the unit mutants of `test_jit_resid`) are those of round sp33 and were not re-run (no change to the generator).

### 3. Counting: `always_inline` on `count_shared_offsets`

After the AVX2 offsets function (round sp36) `count_shared_offsets` was no longer inlined into the tier's shared-memory handler (about 5 % of sortingNetworks' cycles in `perf`). F = `[[gnu::always_inline]] inline` on it (`nm` shows no out-of-line copy in F and one in A, A2 and N). A, A2 (copy of A), F and N (A with a no-op line in the same function, the control for what a layout change does) run in turn, alternating order, 15 rounds (10 for SYR2K and shmembench, 5 for alignedTypes), one job at a time under the lock, `taskset`, `VGPU_THREADS=1`, `nvidia/rtx3060`, SASS, tier on, JIT off; medians of wall seconds:

| program | runs A/A2/F/N | A s (spread) | A2 s (spread) | F s (spread) | N s (spread) | **A/F** | A/A2 | **N/F** | A2/F |
|---|---:|---|---|---|---|---:|---:|---:|---:|
| samples/sortingNetworks | 15 | 9.54 (17%) | 9.21 (34%) | 8.96 (32%) | 9.00 (21%) | 1.064 | 1.036 | **1.005** | 1.028 |
| samples/mergeSort | 15 | 5.84 (2%) | 5.81 (3%) | 5.73 (1%) | 5.77 (34%) | 1.019 | 1.006 | **1.007** | 1.013 |
| samples/histogram | 15 | 9.76 (5%) | 9.68 (5%) | 9.82 (5%) | 9.58 (4%) | 0.994 | 1.008 | **0.976** | 0.986 |
| samples/convolutionSeparable | 15 | 8.10 (6%) | 8.03 (2%) | 7.75 (4%) | 7.99 (3%) | 1.045 | 1.009 | **1.030** | 1.035 |
| polybench/SYR2K | 10 | 7.84 (1%) | 7.83 (1%) | 7.85 (1%) | 7.86 (2%) | 0.998 | 1.000 | **1.001** | 0.997 |
| hecbench/shmembench-cuda | 10 | 3.55 (13%) | 3.56 (4%) | 3.50 (14%) | 3.55 (5%) | 1.015 | 0.997 | **1.013** | 1.018 |
| samples/alignedTypes | 5 | 26.20 (2%) | 26.18 (1%) | 26.26 (3%) | 26.06 (1%) | 0.998 | 1.001 | **0.993** | 0.997 |

Reading it: **not beyond the noise, not kept.** On the three programs the change was for: sortingNetworks A/F 1.064 but A/A2 1.036 (spreads of 17 to 34 %) and N/F 1.005; mergeSort 1.019 with A/A2 1.006 and N/F 1.007; histogram **0.994, with N/F 0.976** (F is the slower build). convolutionSeparable shows 1.045 (N/F 1.030, F against the control build: the one positive row, in a program whose counting is the table's hit path), SYR2K, shmembench and alignedTypes are flat (0.998, 1.015 with a spread of 13 to 14 %, 0.998). Winning on one program and losing on another against the control is the layout lottery of round sp36 (the tier's handlers move when anything in the unit changes) and a function that is inlined into a hot handler grows it; no branch change is carried. No code or test changed for this item; the tree has no `always_inline` on `count_shared_offsets`.

### Caveats
* The budget is time-based, so which launches are recorded differs from run to run and between machines; a program that uploads more than the credit between short launches loses the hits after the upload (`simpleCudaGraphs` loses 5 warm hits of 24). `VGPU_MEMO_EVENT_CREDIT_MS` and `VGPU_MEMO_BUDGET=inf` give the earlier behaviour.
* The programs under 50 ms pay 0.05 to 0.3 ms per process (up to 1.2 ms when they record and write an entry): a double-digit percentage of a 1 to 4 ms run, which no budget can bring down.
* The long programs were measured for 4 of the 13 (the other 9 would each take 30 to 50 minutes of an exclusive server for 3 runs of four states): FDTD-2D is +4.2 % cold (151 of 1,500 launches recorded, the 3 % budget at work).
* The JIT timings were taken on a box with CI runners and production Postgres; spreads are in the table, A/A control programs read 0.994 to 1.009 in the ones that carry a conclusion.
* The e2e shards ran on server2 with programs compiled with nvcc 13.0 elsewhere (no nvcc on the server); the stage `form` was not run under a JIT mutant.

## Round sp39 (release-wide PGO): no speed-up on held-out programs, stays opt-in (branch `sp39-pgo`)

A plain Release build (A), a copy of it (A2, the noise control) and a PGO build (P) trained on 122 manifest entries (395 runs, 179 profile files) were compared on 47 held-out programs the profile never ran (22 NVIDIA, 7 AMD, 12 pantheon workloads, one PyTorch group, 5 tiny programs). Ratios are A/P of median CPU seconds, above 1 means PGO is faster. Server2 under the builder lock, load below 5.

| group (1 thread) | programs | A/A2 | A/P geomean | min | max |
|---|---:|---:|---:|---:|---:|
| NVIDIA | 22 | 0.999 | 0.958 | 0.488 | 1.120 |
| AMD | 7 | 0.999 | 0.958 | 0.884 | 1.017 |
| pantheon workloads (CPU s per 1e9 warp instructions) | 12 | 1.012 | 1.065 | 0.959 | 1.619 |
| tiny start-up | 5 | 0.998 | 1.029 | 1.008 | 1.052 |
| all 47 | 47 | | 0.993 | 0.488 | 1.619 |

At 8 threads (wall, 29 NVIDIA and AMD programs) the geomean is 0.946. The median NVIDIA program is 4% faster, but `shmembench-cuda` (0.57x), `aobench-cuda-ptx` (0.49x) and `immaTensorCoreGemm` (0.76x) lose reproducibly, and five of seven AMD programs are 2 to 12% slower. `fused_attention` at 1.62x is unexplained.

* `-fprofile-partial-training` repairs `immaTensorCoreGemm` (1.056x) and part of `aobench-cuda-ptx`, but makes `test_amd_gcn_same_bits` fail (one `v_mac_f32_e32` result of 16,806 differs), so it was dropped.
* Bits: counters, kernel digests and final memory hashes are equal on all 35 non-pantheon held-out programs at 1 and 8 threads (70 runs), except three runs that already differ between A and A2. The pantheon workloads were not bit-checked (their counters depend on the clock).
* Full ctest on the PGO build: 867 tests, 678 skipped, 8 failed; seven fail the same way on the plain build here. The PGO-only failure is `test_amd_matrix_kernels_layout`: `-fprofile-use` drops the 64-byte alignment of the AVX2 tile kernels.
* Not measured: BOLT (`llvm-bolt` is not installed on server2 or server3, and `perf` is blocked there).
* Caveats: the held-out PyTorch set is one 1 s group, and the cause of the `shmembench` and `radixsort2` losses is a hypothesis.

Verdict: PGO is not enabled for releases. `scripts/pgo-build.sh`, the manifests and `docs/pgo.md` stay as an opt-in tool; `VGPU_PGO=use` now fails on a profile that does not match the source (`VGPU_PGO_STRICT`).


## Round sp40 (store-forwarding stalls in the SIMD kernels, register-file clear at block setup): both measured, neither kept (branch `sp40-simd-stalls`, base `origin/main` d462bc2f)

Question: the sp37 round left two leads: (1) failed store-to-load forwards (`ls_bad_status2.stli_other`: `icmp_mask_k` 15.5 %, `step_warp` 13.5 %, `h_global<false>` 7.6 % of the events) where a kernel loads a 32-byte row the baseline-compiled handlers stored as 16-byte halves; (2) the `rep stosb` clear of the register file at block setup. Keep only what gains 3 % or more pooled.

Setup: server2 (Ryzen 9 9950X, AVX-512 table selected, kernels run as `avx2_impl`), `VGPU_THREADS=1`, cpu 14, one process at a time under the builder lock, whole-process wall medians of 5 interleaved rounds (order alternating), A/A = a second copy of the base library. IBS (`perf record -e ibs_op/cnt_ctl=0/`) works on server2 with sudo. server3 was unreachable, so the pool is the four programs below (mergeSort, LU, SYR2K, interval), not ten: the ceilings measured are small enough that more programs could not change the verdict.

### Lead 1: wide stores for rows the kernels load (variant e1)
Change tried: two new table kernels `copy_row` and `fill_row` (whole 32-byte vector stores), used by `tier_store`'s full-warp row copy, the `SEL` copies, the global-load row copy and the constant-bank/uniform-register broadcasts of `tier_dyn`. Same bits by construction.
* `perf stat`, mergeSort: stli events 893 M / 795 M (base) -> 263 M / 359 M (e1), a 60 to 70 % drop; cycles:u 30.47 G / 30.39 G -> 30.82 G / 30.90 G. LU: stli 9.2 G / 9.0 G -> 4.4 G / 4.5 G; cycles 110.0 G / 109.5 G -> 107.2 G / 110.8 G.
* Wall, medians of 5 (s): mergeSort base 5.83, base2 5.87, e1 6.19; LU 21.22, 21.42, 20.89; SYR2K 7.87, 7.83, 8.00; interval 12.58, 12.66, 12.94. Pooled over the four: base 47.50 s, e1 48.02 s: **0.989x, slower**.
* IBS of the base (mergeSort): `icmp_mask_k` 2.2 %, `ibin_m_k` 2.2 %, `blend_m_k` 2.1 %, `iadd3x_k` 2.1 % of ops; LU `icmp_mask_k` 2.5 %. The kernels' share of time is small, and removing most of the failed forwards moved no cycles: the stalls overlap other work (out-of-order), as sp37's `perf stat` already suggested. The extra indirect call per row costs what it might have saved. Not kept.

### Lead 2: the register-file clear (ceiling variant, no clear at all)
A build that skips `w.r.assign(nregs * 32, 0)` once the vector has its size (wrong in general, a timing ceiling only): mergeSort 5.84 s (base 5.83), LU 20.25 (21.22, **-4.6 %**), SYR2K 7.89 (7.87), interval 12.55 (12.58). Pooled over the four: 46.53 s against 47.50 s, **at most 1.020x with the whole clear removed**; any real change can only drop the slack rows (8 of `regs + 8`, a quarter to 40 %), so about 0.5 % pooled, 1 % on LU. Below the 3 % bar; not built.

No source change is kept in this round; the experiment diff of lead 1 is not committed. A/A (base vs base2): 0.1 to 0.9 % on every program, so only the LU ceiling (4.6 %) and e1's mergeSort loss (6 %) are above noise.

## Round sp41 (counting of memory accesses in the PTX interpreter and the AMD engine): ceilings measured first, all three leads below the bar, nothing built (branch `sp41-ptx-amd-counting`, base `origin/main` d462bc2f)

Question: sp36 gave the SASS path AVX2 kernels for bank passes and sectors. Does the same idea pay (1) in the PTX interpreter's counting (`count_addresses`, `memcount::ptx_bank_passes_*`, `distinct_sectors`), (2) in the AMD engine's LDS bank-conflict and global access counting, (3) as a 32-key AVX2 sort for scatter and duplicate-address handling? Rule set beforehand: measure the share of time in the scalar counting code first, skip a lead whose pooled share is under about 5 %, build only leads with a real ceiling.

Setup: server2 (Ryzen 9 9950X), build of this branch (Release, LTO, `-fno-omit-frame-pointer -g1`), `VGPU_THREADS=1`, cpu 14, `nice -n 19`, one process at a time under the builder lock, load average under 2 throughout. `perf record -e ibs_op/cnt_ctl=0/ -c 200000` (IBS, not `cycles:u`), shares by source line (`perf report --sort srcline`: the innermost inlined line, so a line of `mem_counters.hpp` is counted even when it was inlined into `exec_ld`). Programs are the CUDA Samples built earlier (`~/.cache/cuda-samples`), run with `VGPU_SASS=0` (the PTX interpreter), `VGPU_GPU=nvidia/rtx3060`. server3 did not answer (no route to host).

### Lead 1: PTX interpreter counting (share of IBS samples, lines of `include/vgpu/exec/mem_counters.hpp` and the counting functions of `interpreter.cpp`)

| program | wall s (IBS) | bank miss (`ptx_bank_passes_table`) | memo hit path | sector list (`distinct_sectors` fallback) | sector closed forms | all of mem_counters.hpp | counting in interpreter.cpp |
|---|---:|---:|---:|---:|---:|---:|---:|
| histogram | 35.3 | 5.79 % | 0.61 % | 8.35 % | 0.24 % | 15.04 % | 0.99 % |
| sortingNetworks | 29.2 | 2.49 % | 9.95 % | 1.35 % | 1.17 % | 15.32 % | 1.88 % |
| mergeSort | 15.2 | 3.41 % | 0.66 % | 4.17 % | 1.52 % | 9.98 % | 1.05 % |
| convolutionSeparable | 22.2 | 0.22 % | 3.77 % | 0.50 % | 1.71 % | 6.35 % | 1.42 % |
| transpose | 16.3 | 0.14 % | 0 | 0.17 % | 1.60 % | 2.24 % | 0.79 % |
| reduction | 12.4 | 0 | 0 | 0.19 % | 0 | 0.27 % | 0.30 % |
| dct8x8 | 3.3 | 0.23 % | 1.43 % | 0.36 % | 2.47 % | 4.90 % | 1.43 % |
| scalarProd | 2.3 | 0 | 0 | 0.11 % | 0.54 % | 0.65 % | 0.22 % |
| matrixMul | 103.2 | 0.48 % | 0 | 0.02 % | 5.12 % | 6.14 % | 1.87 % |
| scan (stopped at 240 s) | 241 | 0.28 % | 0.63 % | 0.05 % | 2.50 % | 3.96 % | 1.14 % |

Pooled over the ten (wall-weighted / plain mean of shares): bank-miss path 0.94 % / 1.30 %, memo hit path 1.17 % / 1.71 %, sector list 0.89 % / 1.53 %, sector closed forms 2.64 % / 1.69 % (matrixMul and scan; already arithmetic), **all counting 6.07 % / 6.48 %**. Even deleting every counting instruction would give at most 1.065x pooled; each separate lever is 1 to 2.6 % pooled. A lever would need to remove its whole share to approach the 3 % bar and cannot. The programs with large shares (histogram 14 %, sortingNetworks 15 %, mergeSort 10 %) are four of ten; they could gain a few percent each, but the pooled gain over six or more programs stays under 2 % in the best case (taking the bank-miss, memo and sector-list shares of those four to zero would be about 2.4 % wall-weighted over the ten, and no kernel removes all of it). Not built. (Mapping that was worked out, for a later round: a PTX lane's words are `addr>>2 .. addr>>2 + ceil(span/4)`, which is the SASS formulation with the address rounded down to a word and `bytes = ceil(span/4) * 4`, so `sass_bank_passes_widened` and `simd::bank_window_kernel` could serve `ptx_bank_passes_table` for 32 lanes, spans up to 16 bytes, with all active addresses sharing their upper 32 bits.)

### Lead 2: AMD engine LDS bank-conflict and global access counting: nothing to speed up
The AMD engine has no bank-conflict or sector counters at all: `grep -i 'bank_conf\|sector\|shared_requests\|global_requests'` over `amd/src` finds none outside the HIP API's text (the GCN executor counts instructions, not memory transactions). The only scalar address work on the memory path is the distinct-address filter of the global-load handler (`gcn_exec.cpp` near line 10217, a few lines that drop lanes sharing an address). IBS of `bench_amd` (gfx1030 on `amd/rx6800`, `full` size; mi300x lds) over `mem`, `lds`, `ldsx`, `flat`, `narrow`, `atomics` and `lds` on mi300x: those lines are 0.40 % of the samples for `mem` and 0.00 % for every other workload; the whole of `gcn_exec` + tier + memory manager is 76 to 91 % of the samples (50 % for `atomics`), so there is no counting code in them to vectorize. Ceiling 0 %, not worth building. (Caveat: the workloads' device phases are short, 5 to 180 ms, so process start-up is part of the rest; the line shares of the device phase are the ones above divided by the share of the program the device phase is, which cannot lift 0.00 % to anything.)

### Lead 3: 32-key AVX2 sort for scatter and duplicate addresses
Nothing in the PTX counting sorts: `distinct_sectors` keeps a list and searches it newest first (sp36 and earlier measured a sort as slower than the search), and `ptx_bank_passes_table` uses the per-bank table. The share the sort would replace is the sector-list column above: 0.89 % wall-weighted, 1.53 % mean (histogram 8.35 %, mergeSort 4.17 %, the rest under 1.4 %). Below the bar; not built. The AMD distinct-address filter is 0.40 % on one workload. Not built.

### Result
No source change is kept: the branch carries only this section. No timings against a control copy were taken because nothing was built (so there is no A/A figure to report; the IBS runs are single profiles, one per program, not repeated). Mutation tests and a differential test are therefore not added. Cleanup: the scratch directories of this round on server2 were removed by exact path.


## Round sp42 (host-GPU measured on A10G)

Question: what does `VGPU_HOST_GPU=exact` (PR #529 hooks, main d462bc2f) buy on a real card at today's main, program by program, and what does it refuse or lose? Earlier rounds measured six programs (sp14/sp15: FDTD-2D, babelstream, base64e, aligned-types, LU, GRAMSCHM) and replayed 64 traces; this round runs a wider set in one session, with the native card as a third column. Nothing in the source changed; the mode, its constants and its gates are the ones of d462bc2f.

Setup: AWS g5.xlarge (A10G, 4 vCPU, driver 595.112.07, Ubuntu 24.04), simulator built from a clean worktree of d462bc2f, `RelWithDebInfo`, no LTO, g++ 13. The programs are the cached sm_86 binaries from `~/.cache` (PolyBench/GPU, Rodinia with the round-12 arguments, HeCBench, `cudart` shared, CUDA 13.0) plus one synthetic program, `data/sp42/sens.cu` (integer-only kernels, so every kernel is in the exact set). Three modes per program, same binary, one process each: **native** (real `libcudart` 13.0, the card), **sim** (`VGPU_GPU=nvidia/a10g VGPU_THREADS=4 VGPU_QUIET=1`, shim `libcudart`), **host-GPU** (the same plus `VGPU_HOST_GPU=exact VGPU_HOST_GPU_LOG=1`). Wall seconds of the whole process (`measure.sh`, `batch.sh`, `batch2.sh`, `results.tsv` and the host-GPU logs are in `docs/perf-design/data/sp42/`). One run per cell, no repeats: differences under about 0.05 s on the short rows are noise, and the 600 s cap cut the simulator short on aligned-types (`+` in the table; the 2,682 s of an earlier round is the uncapped figure). Speedup = sim / host-GPU wall.

How to read the PolyBench rows: those programs also run a CPU reference computation, so the process wall hides the device phase (2MM: native 187 s, of which the card is 0.05 s). The "device phase" column is the number the program itself prints as `GPU Time in seconds` (sim, host-GPU; native is 0.0004 to 2.6 s and is in the raw outputs, not tabled: its timer covers the asynchronous enqueue).

"stdout host vs sim" compares standard output with every line that carries a time, rate or bandwidth removed. "verify" is `VGPU_HOST_GPU_VERIFY=1` (every launch the mode would offload runs on the card and then in the simulator, memory compared): run on the Rodinia and HeCBench rows only, 150 s cap. It was not run on the PolyBench rows because it did not finish (see finding 6).

| program | native s | sim s | host-GPU s | sim / host | device phase sim / host s | launches card / sim | stdout host vs sim | verify (in-set launches same / diff) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| poly/2DCONV | 0.82 | 0.97 | 0.89 | 1.09x | 0.39 / 0.18 | 1 / 0 | same | not run |
| poly/3DCONV | 0.49 | 0.77 | 0.91 | 0.85x | 0.50 / 0.56 | 0 / 254 | same | not run |
| poly/3MM | 1.50 | 2.45 | 1.48 | 1.66x | 1.19 / 0.15 | 3 / 0 | same | not run |
| poly/GEMM | 0.58 | 0.82 | 0.67 | 1.22x | 0.44 / 0.14 | 1 / 0 | same | not run |
| poly/ATAX | 0.33 | 1.09 | 1.22 | 0.89x | 0.99 / 1.03 | 0 / 2 | same | not run |
| poly/MVT | 0.48 | 1.23 | 1.37 | 0.90x | 0.99 / 1.03 | 0 / 2 | same | not run |
| poly/GESUMMV | 0.42 | 0.39 | 0.54 | 0.72x | 0.19 / 0.24 | 0 / 1 | same | not run |
| poly/JACOBI2D | 0.29 | 0.64 | 0.36 | 1.78x | 0.59 / 0.21 | 36 / 4 | same | not run |
| poly/COVAR | 63.20 | 67.21 | 71.26 | 0.94x | 27.01 / 27.12 | 0 / 3 | same | not run |
| poly/CORR | 42.25 | 67.42 | 67.17 | 1.00x | 27.08 / 27.16 | 0 / 4 | same | not run |
| poly/ADI | 0.33 | 0.41 | 0.41 | 1.00x | 0.33 / 0.34 | 0 / 2049 | same | not run |
| poly/GEMVER | 0.47 | 0.67 | 0.68 | 0.99x | 0.43 / 0.34 | 1 / 2 | same | not run |
| poly/BICG | 0.36 | 0.27 | 0.40 | 0.68x | 0.13 / 0.16 | 0 / 2 | same | not run |
| poly/JACOBI1D | 0.45 | 2.48 | 2.76 | 0.90x | 2.43 / 2.62 | 0 / 20000 | same | not run |
| rod/backprop | 0.25 | 0.05 | 0.27 | 0.19x | - | 1 / 1 | same | 1 / 0 |
| rod/gaussian | 0.24 | 0.02 | 0.24 | 0.08x | - | 61 / 65 | same | 63 / 0 |
| rod/lud | 0.24 | 0.05 | 0.27 | 0.19x | - | 10 / 36 | same | 15 / 0 |
| rod/needle | 0.23 | 0.02 | 0.26 | 0.08x | - | 14 / 17 | same | 31 / 0 |
| rod/pathfinder | 0.24 | 0.01 | 0.24 | 0.04x | - | 1 / 2 | same | 3 / 0 |
| rod/hotspot | 0.24 | 0.03 | 0.03 | 1.00x | - | 0 / 10 | same | 0 / 0 |
| rod/bfs | 0.24 | 0.02 | 0.25 | 0.08x | - | 0 / 26 | same | 26 / 0 |
| rod/srad_v2 | 0.24 | 0.01 | 0.01 | 1.00x | - | 0 / 1 | same | 0 / 0 |
| hec/matrixT-cuda | 0.24 | 0.06 | 0.27 | 0.22x | - | 8 / 1 | same | 8 / 0 |
| hec/mandelbrot-cuda | 0.53 | 1.20 | 1.20 | 1.00x | - | 0 / 2 | same | 0 / 0 |
| hec/md5hash-cuda | 0.29 | 153.93 | 153.88 | 1.00x | - | 0 / 4 | same | not run |
| hec/memcpy-cuda | 0.37 | 0.04 | 0.04 | 1.00x | - | no kernel | same | not run |
| hec/mask-cuda | 0.39 | 0.14 | 0.15 | 0.93x | - | 0 / 6 | same | 0 / 0 |
| hec/maxpool3d-cuda | 0.25 | 0.01 | 0.24 | 0.04x | - | 0 / 1 | same | 1 / 0 |
| hec/merge-cuda | 0.25 | 0.07 | 0.24 | 0.29x | - | 0 / 16 | same | 8 / 0 |
| hec/lud-cuda | 0.24 | 0.05 | 0.26 | 0.19x | - | 10 / 36 | same | 15 / 0 |
| hec/matrix-rotate-cuda | 0.23 | 0.01 | 0.01 | 1.00x | - | 0 / 1 | same | 1 / 0 |
| hec/lombscargle-cuda | 1.53 | 3.04 | 3.04 | 1.00x | - | 0 / 1 | same | 0 / 0 |
| poly/SYRK | 1.27 | 5.17 | 1.28 | 4.04x | 4.13 / 0.16 | 1 / 0 | same | not run |
| poly/SYR2K | 2.30 | 9.80 | 2.32 | 4.22x | 7.75 / 0.17 | 1 / 0 | same | not run |
| poly/LU | 1.12 | 31.41 | 11.42 | 2.75x | 30.59 / 10.49 | 1407 / 2689 | same | not run |
| poly/2MM | 187.08 | 183.18 | 122.60 | 1.49x | 59.40 / 0.23 | 2 / 0 | same | not run |
| poly/FDTD-2D | 4.26 | 110.28 | 4.75 | 23.22x | 106.41 / 0.82 | 1494 / 6 | same | not run |
| hec/base64e-cuda | 3.18 | 320.91 | 8.95 | 35.86x | - | 514 / 2 | timing/bandwidth lines only | not run |
| hec/babelstream-cuda | 3.13 | 405.24 | 13.66 | 29.67x | - | 1188 / 26 | timing/bandwidth lines only | not run |
| hec/aligned-types-cuda | 3.79 | 600.02+ | 8.61 | >70x | - | 12974 / 26 | (sim hit 600 s cap) host == native | not run |
| poly/GRAMSCHM | 230.62 | 272.54 | 277.58 | 0.98x | 49.75 / 44.44 | 2046 / 4098 | same | not run |
| hec/adam-cuda | 4.92 | 26.25 | 26.23 | 1.00x | - | 0 / 100 | same | 0 / 0 |
| hec/axpby-cuda | 31.04 | 202.19 | 219.66 | 0.92x | - | 0 / 12770 | same | not run |

| synthetic (integer kernel) | launches | elements | native s | sim s | host-GPU s | sim / host | launches card / sim | final-memory checksum |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| compute | 1000 | 4,096 | 0.25 | 2.03 | 0.26 | 7.81x | 998 / 2 | same |
| compute | 100 | 65,536 | 0.23 | 3.14 | 0.26 | 12.08x | 98 / 2 | same |
| compute | 10 | 1,048,576 | 0.25 | 4.99 | 1.26 | 3.96x | 8 / 2 | same |
| compute | 4 | 4,194,304 | 0.26 | 8.00 | 2.30 | 3.48x | 3 / 1 | same |
| compute | 1 | 1,048,576 | 0.24 | 0.51 | 0.26 | 1.96x | 1 / 0 | same |
| compute | 1 | 4,096 | 0.23 | 0.01 | 0.25 | 0.04x | 0 / 1 | same |
| mem | 2000 | 4,096 | 0.24 | 0.23 | 0.32 | 0.72x | 0 / 2000 | same |
| mem | 200 | 262,144 | 0.24 | 0.57 | 0.25 | 2.28x | 198 / 2 | same |
| mem | 50 | 4,194,304 | 0.26 | 2.24 | 0.41 | 5.46x | 48 / 2 | same |
| mem | 20 | 16,777,216 | 0.32 | 3.63 | 0.83 | 4.37x | 18 / 2 | same |
| mem | 1 | 16,777,216 | 0.32 | 0.28 | 0.41 | 0.68x | 1 / 0 | same |
| mem | 4 | 4,194,304 | 0.26 | 0.21 | 0.39 | 0.54x | 2 / 2 | same |

### Results

1. **Gains need a large kernel in the exact set, and they are large when they exist.** FDTD-2D 23x (110.3 s to 4.75 s; the native card takes 4.26 s, so the mode is within 12% of the card), base64e 36x, babelstream 30x, aligned-types over 70x (sim stopped at 600 s; 8.6 s on the card, native 3.8 s), SYRK 4.0x, SYR2K 4.2x (1.28 s and 2.32 s against native 1.27 s and 2.30 s: the card path costs the 0.2 s start-up and nothing else). On the device phase alone: 2MM 59.4 s to 0.23 s, FDTD-2D 106 s to 0.82 s, SYR2K 7.75 s to 0.17 s. LU 2.75x (31.4 s to 11.4 s), JACOBI2D 1.78x, 3MM 1.66x, 2MM 1.49x on the process wall.
2. **Everything that matched did match.** Stdout equals the simulator's in all 42 rows where the simulator finished, apart from bandwidth and elapsed-time lines in base64e and babelstream (PASS in all three modes). Final memory: the 12 synthetic runs print a checksum of the final device buffer and it is equal in all three modes. `VGPU_HOST_GPU_VERIFY=1` compared 172 in-set launches on the Rodinia and HeCBench programs (all kernels that ran on the card or would have): 0 differ. PolyBench's own CPU-vs-GPU check reports 0 mismatches in every row. Nothing was run outside the exact set (`VERIFY=all` not used).
3. **The first card launch costs about 0.23 s, and a program that finishes in less than that loses.** The native process of a trivial program also takes 0.24 s (driver start), the simulator's 0.01 to 0.05 s. All eight Rodinia programs and several HeCBench ones sit there: host-GPU is 0.24 to 0.27 s against 0.01 to 0.07 s simulated (0.04x to 0.29x), whether or not the launch ended on the card (bfs, maxpool3d, merge and the synthetic one-launch 4,096-element row kept every launch in the simulator and still paid it; matrix-rotate did not: the cause of the difference is not investigated). The size rule compares one launch against its simulator time and does not charge the one-off start-up. That is the only systematic slowdown found: under 0.3 s of simulated time, leave the mode off. No row of a program that runs longer than 0.5 s in the simulator got slower than 0.68x, and the 0.68x to 0.72x cases (BICG, GESUMMV, synthetic mem single-launch and tiny-launch rows) are 0.1 to 0.2 s absolute.
4. **Size and launch-count sensitivity** (synthetic table above; the kernels are integer-only; compute = 256-iteration xorshift/IMAD loop per thread, mem = grid-stride read-multiply-write):
   * Many small launches win once a kernel has a history: 1,000 launches of 16 blocks, 7.8x; 100 launches of 256 blocks, 12.1x. The first two launches of a kernel run in the simulator (history rule), the rest (998 and 98) go to the card.
   * One launch is not enough: compute 1 x 1M threads is 1.96x (first-launch static rule offloaded it), compute 1 x 4,096 threads and mem 1 x 16M words stay at or below 1x (0.04x and 0.68x: start-up plus a 128 MiB push for a 0.28 s simulator launch).
   * Big buffers and few launches give a smaller gain: mem 50 x 4M words 5.5x, mem 20 x 16M words 4.4x (256 MiB pushed, 192 MiB pulled back), compute 10 x 1M threads 4.0x and 4 x 4M threads 3.5x. The launches a kernel runs in the simulator before it has a history (2 of 10, 1 of 4 here) are simulated at full price, which bounds the gain when a big kernel is launched only a few times.
   * Tiny memory kernels never leave the simulator (mem 2,000 launches of 4,096 words: sim estimate below the card path), and that costs 0.09 s of start-up (0.72x).
   * The program that shows both ends: LU. 1,407 launches on the card and 2,689 in the simulator, 22.5 GiB pushed and pulled in each direction; one kernel (`lu_kernel2`) was banned by the traffic rule after two launches that pushed 16 MiB of 16 MiB live; 639 launches too small, 2,048 refused (MUFU). 2.75x here against native 1.12 s.
5. **What was refused, and why** (`VGPU_HOST_GPU_LOG=1` refusal lines; the per-program logs were trimmed from the repo and the results are in `data/sp42/results.tsv`):
   * `MUFU.RCP` / `MUFU.RSQ` outside the division idiom: Rodinia backprop, lud, hotspot, srad_v2; PolyBench ADI, LU kernel 1, GRAMSCHM (RSQ for `sqrt`, RCP), CORR `reduce_kernel`; HeCBench adam, mask, md5hash, mandelbrot, matrixT (`transposeDiagonal`), lud. By far the most common reason in this corpus. The IEEE-division recognizer (`VGPU_HOST_GPU_DIV=1`) is not on by default and was not measured this round.
   * `CALL.REL.NOINC` (the slow path of division or `sqrt`, not recognised without the flag): COVAR and CORR `mean_kernel` / `std_kernel`, HeCBench merge `workloadDiagonals`.
   * "too small" (size rule says the card path costs more than the simulator): axpby (12,768 of 12,770 launches: 5 ms of simulator time against a 520 ms card path because 5.9 GiB are live), a part of LU (639), Rodinia lud (3). md5hash simulates in 154 s and is refused for MUFU, so the 154 s are not reducible by this mode.
   * Not seen this round: texture/surface, cluster, cooperative, non-32 warp, VMM/IPC/heap launches (`card_blocker`), float atomics, tensor-core MMA (no program here exercised them; unit and e2e tests cover the refusal).
6. **Finding, not fixed (verify mode only):** `VGPU_HOST_GPU_VERIFY=1` on PolyBench 2DCONV (128 MiB live) ran for more than 8 CPU-minutes (killed; plain simulation: 0.97 s). `perf` on the stuck process: 35% in `vgpu::hostgpu::Runtime::Impl::before_access` (a `std::count_if` over the chunk `Where` vector, per access) and 43% in kernel `_raw_spin_unlock_irqrestore`: the simulator's accesses to card-valid memory ask the gate per access and the gate scans linearly. It does not affect the normal mode (the simulator runs after `before_simulated` has pulled the card copy back), only verify runs with a large resident allocation, which is why `verify_ban.sh` of sp14 used a 150 s cap. The PolyBench verify rows are therefore missing; the 172 verified launches are the small-memory programs.

### What the mode does and does not preserve (from 08, 09, 10 and the code; the measurements above agree)

Preserved for a program that runs in this mode: the bits of every result of the kernels it offloads, defined by the exact set (integer arithmetic, IEEE `FADD/FMUL/FFMA/FSETP`, conversions with defined rounding, plain loads and stores, integer reductions with no return value, barriers, plain control flow; no MUFU, no float or compare-and-swap atomics, no tensor-core MMA), checked by the simulator's own bits on 172 launches here and on 21,804 + 23,248 in earlier rounds; memory contents at the simulator's device addresses (the card holds a copy at the same addresses and gives chunks back before any simulator access: every access path of `MemoryManager` asks the gate, 10 section 2); the API semantics (allocation, copies, events, graphs, libraries stay the simulator's; offloaded launches are synchronous like the simulator's); stdout, exit codes and files. The simulated profile must be the host card (`nvidia/a10g` on an A10G) and the module must carry SASS for it.

Not preserved, for the launches that run on the card (08 section 5, 10 section 3): performance counters and profiles, per-instruction traces and SASS trace, kernel digests, CUPTI activity and callback records of those kernels, the race detector, strict mode, fault injection, the deterministic scheduler's interleaving, out-of-bounds and use-after-free diagnostics, `VGPU_MAX_STEPS`, the debugger. The mode refuses to start when counters, trace, race, strict, faults, a non-default scheduler or digests are switched on (it says so, also under `VGPU_QUIET`), so these are not silently dropped; the debugger and CUPTI subscribers have no switch it can test. Not detectable statically and therefore a stated limit: data races, reads of uninitialised memory and out-of-bounds reads the card tolerates (Rodinia `srad_v2` is the example: native and simulator differ in this session's run, the mode refused its kernel and so equals the simulator); a pointer stored in device memory that points into pinned or managed host memory; another host thread launching simulator kernels while an offloaded launch ends. Launches are refused while a VMM mapping, an IPC import or a kernel-heap block exists. Everything the exact set leaves out runs in the simulator, so a program's result is the simulator's whichever kernels went to the card.

### Verdict

On an A10G the mode reaches the native card's time on kernels it can run (FDTD-2D within 12%, SYRK/SYR2K within 1%), gives 4x to over 70x on programs whose simulator time is dominated by an eligible kernel, matches the simulator's output everywhere it was compared, and costs nothing when no kernel is eligible. It loses, by about 0.23 s, on programs shorter than that, and gains nothing on the most common kernel shape of this corpus, anything with a division or square root (the `MUFU` family), which `VGPU_HOST_GPU_DIV=1` would admit for the division case only. Candidate follow-ups, not done: charge the start-up in the size rule (or start the driver on first offload only), measure `VGPU_HOST_GPU_DIV=1` on the division-bound programs listed above, and make the gate's scan constant-time so verify mode is usable on large allocations.


## Round sp43

Branch `sp43-hostgpu-fixes` (base `origin/main` d462bc2f). Three findings of Round sp42 (host-GPU mode on an A10G), fixed or analysed with no card: the 0.23 s first-launch cost, a verify run that never finished, and the MUFU refusals. Everything below ran on server2 (Zen 5, `-j3`, `nice -n 19`, one builder under `flock`, load 3 to 12 from the other jobs); no card, no AWS resource was used. Nothing changed for a run with `VGPU_HOST_GPU` unset: the sources touched are `src/runtime/host_gpu.cpp` (reached only through `Runtime::for_device`, which returns before anything else when the mode is off) and headers included only by it, `runtime.cpp` (through `host_gpu_rt.hpp`) and the tests.

### 1. The first-launch cost (the driver starts only when a launch has been decided to go)

Cause, read from the code and reproduced (below). `try_launch` called `Impl::begin_start()` at the FIRST launch of any kernel in the exact set on a grid of 32 warps or more, before the size rule had looked at the launch; the start runs on a thread (the driver load, `cuInit`, the primary context, the 1 TiB reservation: 0.23 s on the A10G), and the `atexit` hook `settle_start()` waits for that thread. So a program with one such launch paid the whole start at exit even when every launch stayed in the simulator: Rodinia `bfs` (26 launches, 22 of them "too small": sim 0.1 ms against a card path of 0.12 ms), HeCBench `maxpool3d`, `merge` (its in-set kernel; the refused `workloadDiagonals` is not what started it). `matrix-rotate` did not pay because its single launch is under 32 warps, the old start condition (a stand-in-driver run of main shows `cuInit` called for every program of the table below except `needle` and `matrix-rotate`). Nothing else touched the driver early: `for_device` only reads switches, `card_blocker()` and the residency/`CardGate` bookkeeping never call it, the `atexit` hook only waits for a start that was made.

Change (`include/vgpu/host_gpu.hpp`, `src/runtime/host_gpu.cpp`):
* `Costs::start_cost_s = 0.25` (the measured 0.23 s, rounded up; `VGPU_HOST_GPU_START_COST=<seconds>` overrides, `0` is the rule of main: nothing charged).
* `should_offload_with_start(estimate, costs, driver_started, eligible_sim_spent_s)` replaces `should_offload` in the launch decision and in the first-launch rule. With the driver up it is the old rule. Before that, a launch goes to the card when (a) it repays the start alone: `sim_s > margin * (start + card path)`, i.e. the old rule with the start added to the card path (one large launch: SYRK, SYR2K, 2MM, FDTD-2D), or (b) the old rule holds AND the program has already spent at least `start_cost_s` of simulator time in launches of exact-set kernels that stayed in the simulator. (b) is the ski-rental rule: the start is bought once as much has been spent renting as buying costs, so a start that turns out useless costs at most the time already spent (never more than doubling the program) and a program that never spends that much never loads the driver.
* `begin_start()` is called from `await_start()` only, that is when a launch has been decided to go; the early call and `start_worthwhile` (and its unit test) are removed. `note_simulated` adds the time of an in-set simulator launch to the account. No change to what the program sees: the summary line, `VGPU_HOST_GPU_LOG` lines (one more word: "without the driver start" when the start is not paid yet), outputs and exit codes.

Measured (server2, the cached sm_86 binaries of PolyBench/Rodinia/HeCBench, `VGPU_GPU=nvidia/a10g`, the stand-in driver of `nvidia/tests/e2e/host_gpu_stub_driver.c` with `cuInit` sleeping 230 ms, whole-process wall in seconds, one run each; the stand-in refuses to load a module so a launch that reaches the card falls back to the simulator and the output is the simulator's):

| program | mode off | main, mode on | now, mode on | `cuInit` calls main / now |
|---|---:|---:|---:|---|
| rod/bfs | 0.01 | 0.23 | 0.01 | 1 / 0 |
| rod/backprop | 0.01 | 0.25 | 0.02 | 1 / 0 |
| rod/gaussian | 0.00 | 0.23 | 0.00 | 1 / 0 |
| rod/pathfinder | 0.00 | 0.23 | 0.00 | 1 / 0 |
| rod/needle | 0.01 | 0.00 | 0.00 | 0 / 0 |
| hec/maxpool3d | 0.00 | 0.23 | 0.00 | 1 / 0 |
| hec/merge | 0.02 | 0.24 | 0.02 | 1 / 0 |
| hec/matrixT | 0.02 | 0.24 | 0.01 | 1 / 0 |
| hec/matrix-rotate | 0.00 | 0.00 | 0.00 | 0 / 0 |

That is the sp42 table (0.24 to 0.27 s against 0.01 to 0.07 s) reproduced with a stand-in driver and removed. With `VGPU_HOST_GPU_START_COST=0` the new binary behaves as main (backprop 0.25 s, one `cuInit`). The programs the mode helps are unchanged by construction (a launch whose simulator time is above `margin * (0.25 s + path)` starts the driver as before; the loss is that a program of many 5 ms launches, where main started the driver at the first launch, now starts it after 0.25 s of them): not measured on a card, so the gain of base64e, babelstream and aligned-types (hundreds of launches of 1 to 20 ms each: the driver now starts after about 0.25 s of simulation, roughly 0.25 s later than before and the first two launches of a kernel were simulated anyway) is expected to move by that 0.25 s and no more, by the rule above; the first A10G session should confirm it with `VGPU_HOST_GPU_START_COST=0` as the A/B.

Tests: `test_host_gpu_logic` (the rule: the constant, a launch that cannot repay the start, one large launch, the account's threshold, no send of a launch the plain rule refuses, cost 0 equals the old rule, driver up equals the old rule, the start inside `card_path_seconds`, the first-launch rule paying the start), `e2e_host_gpu_stub` steps 7 and 8 (runs everywhere nvcc exists; ran here on server2 with nvcc 13.0: with a start cost of 1000 s neither the history nor the first-launch rule calls the driver; with a cost of half the first launch's time of `acct` (32 times the work of `big`) the account starts it). Steps 2 to 5 now say `VGPU_HOST_GPU_START_COST=0` where they need the card path in a program that runs for milliseconds.

### 2. Verify mode: the per-access scan and the lock

Cause. `Impl::before_access` is called for every simulator access to an allocation with a valid card copy (a verify run keeps the card's copy valid through the simulator's launch, so every access of that launch comes here). Each call took an exclusive `std::mutex`, looked the allocation up and, after settling the accessed chunks, called `update_slot`, which counted every chunk of the allocation twice (`Residency::card_valid()` = two `std::count` over the chunk vector): the 35% of sp42, and the mutex was the 43% of kernel time (four simulator threads waiting on one futex). 2DCONV with 128 MiB live is about 1.6e8 accesses.

Change. (a) `Residency` keeps the number of chunks per state, updated by every change of a chunk: `count()` and `card_valid()` are O(1); `host_chunks()` returns at once when no chunk is the simulator's. (b) `access_changes_state(first, last, write)`: a read changes nothing unless a chunk of it is the card's alone, a write nothing unless every chunk of it is the simulator's alone; (c) the gate's logic moved to `gate_access` in `include/vgpu/host_gpu_track.hpp` (so a unit test runs the real function): it takes a shared lock (`std::shared_mutex`, was `std::mutex`), looks at the accessed chunks only and returns when (b) says nothing changes; only an access that changes something retakes the lock exclusively and decides again. Verify semantics are unchanged: same states, same pulls, same `update_slot`.

Measured on server2 (stand-in driver in `STUB_CARD_OK=1` mode: the module loads and a launch "succeeds" doing nothing, so verify believes it ran on the card and compares; the program's result is the simulator's whatever the card did), PolyBench 2DCONV, `VGPU_HOST_GPU=exact VGPU_HOST_GPU_VERIFY=1`, 4 threads:

| | wall | device phase printed by the program |
|---|---:|---:|
| main | 102.0 s | 101.6 s |
| now | 7.0 s | 6.6 s |

(sp42: more than 8 CPU-minutes, killed; plain simulation 0.97 s on the A10G rig.) 14.5x. The remaining distance to plain simulation is the memory manager's slow path, not the gate: while any allocation is card-valid `resolve()` replaces the threads' lookup memories, so every access takes the table lock and a lookup before the gate's shared lock; changing that is a hot-path change in `MemoryManager` and is not done here. `perf` is not allowed on server2, so the split of the 7 s is not profiled. Unit benchmark of the gate alone (`test_host_gpu_track`, 4 threads x 40,000 reads of a 128 MiB allocation, all chunks card-valid): 0.006 s now against 0.113 s for the old exclusive mutex plus the double count (20x; the test requires 5x).

Tests: `test_host_gpu_logic`: `residency_counts_match_a_recount_after_every_operation` (300 random sequences over every mutator, counts equal a recount after each step), `access_changes_state_is_exactly_whether_the_gate_would_do_anything` (the fast path skips exactly the accesses the full path leaves alone, compared with a copy that really runs `sim_read` / `sim_write`), `residency_queries_do_not_grow_with_the_allocation` (200,000 queries on 2^20 chunks under 2 s; the scan took 20+ s); `test_host_gpu_track`: the settled access asks for nothing, a read pulls once and shares, a write drops the copy and pulls only a partial write of a card-only chunk, accesses outside every entry are ignored, four threads reading 512 card chunks pull each exactly once, four threads mixing reads and writes leave counts and the published slot consistent, and the ratio test above.

### 3. MUFU refusals: why, what a correct fix is, what has to wait for a card

Why `MUFU` is refused. The card's `MUFU.RCP/RSQ/SQRT/EX2/LG2/SIN/COS` are table-and-polynomial approximations (rcp, sqrt, rsq, ex2 within 1 ulp of the exact value, lg2 absolute error near 1, sin/cos/tanh further) and the simulator computes the correctly rounded value in double (`host_gpu_ref.hpp`, `src/simd/kernels.inc mufu_k`, `nvidia/docs/sass.md` "What stays different from the card"). Measured on a card in earlier rounds: `rcp` differs from the simulator's by one ulp on 13% of the mantissas of [1, 2), and every sampled kernel that held one of these and differed was a kernel whose result reaches memory through an approximation (Round 13: silu, perplexity, binomial, aobench, attention; 1 to 171 ulp). A bare `MUFU` therefore cannot be in the exact set, and cannot be made so without reproducing the card's tables in the simulator (2^23 entries per function, not measured; it would change every non-host-GPU result of every kernel with an `expf`, `sqrtf`, `rsqrtf`, so it is not a change for this branch).

What the refused kernels actually hold. New diagnostic: `VGPU_HOST_GPU_LOG=1` now also prints, for a kernel refused for an instruction outside the set, its MUFU forms and counts of FCHK and CALL (`LOG=2` prints the kernel's SASS). Run with the stand-in driver on server2 over the programs of the sp42 list, with and without `VGPU_HOST_GPU_DIV=1` (29 refused kernels; this classification is read from the SASS the simulator decodes, no card):

| what the kernel's MUFUs are | kernels | `VGPU_HOST_GPU_DIV=1` | would be exact because |
|---|---|---|---|
| IEEE float division only (`MUFU.RCP` + FFMA refinement + `FCHK` + a `CALL` to the slow path, whose `MUFU.RSQ R, -QNAN` is the 0/0 NaN) | PolyBench ADI kernels 1 to 6, LU kernel 1, GRAMSCHM kernel 2, CORR `mean_kernel` and `reduce_kernel`; HeCBench mandelbrot, lud `lud_perimeter`; Rodinia lud `lud_perimeter` (13) | **admitted already** | `div.rn.f32` is IEEE on the card (PTX ISA) and the simulator's execution of the idiom equals the correctly rounded quotient for all 2^32 divisors (sp15); the card's `div` run is still pending |
| 32-bit integer `/` and `%` by a run-time value (`I2F.RP`, `MUFU.RCP`, `IADD3 +0xffffffe`, `F2I.FTZ.U32.TRUNC.NTZ`, Newton step, two corrections), no `FCHK`, no `CALL` | HeCBench mask (six kernels), matrixT `transposeDiagonal`, md5hash (plus one unrelated `CALL` to a trap-like target); Rodinia backprop `bpnn_layerforward` (four divisions by 2, 4, 8, 16 that ptxas did not reduce) (9) | refused ("a MUFU.RCP with no FCHK") | integer division is exact on the card by definition; see "integer division" below |
| float division plus IEEE `sqrtf` (`FSETP.GEU |x|, FLT_MIN`, `MUFU.RSQ`, FFMA refinement, a `CALL` to the slow path, no `FCHK` before it) | HeCBench adam; PolyBench CORR `std_kernel`, GRAMSCHM kernel 1 (sqrt only); Rodinia hotspot (4) | refused ("a CALL with no FCHK before it") | `sqrt.rn.f32` is IEEE on the card; needs its own recognizer and the card's `sqrtrn` run |
| float division plus a second call that is not the division slow path | HeCBench lud and Rodinia lud `lud_diagonal` (2) | refused ("the subroutine calls ...") | not analysed (the second subroutine is not read here) |
| double division (`MUFU.RCP64H`) with float division and calls | Rodinia srad_v2 `srad_cuda_1` (1) | refused | needs a double-division recognizer; srad_v2 also reads out of bounds (sp42), so it is the simulator's anyway |

So "MUFU outside the division idiom" in sp42 is 13 kernels the already-written recognizer admits (the flag is off by default and was not measured), 9 integer divisions, 4 square roots, 3 others. What is on the table for the default exact set, with the evidence each needs:

* **Float division (13 kernels): turn `VGPU_HOST_GPU_DIV` on by default.** Needs the card run `hostgpu_sfu_exhaustive div` (12 dividends, all 2^32 divisors, 0 different), then `VERIFY=all` with the flag over the programs above. Not done: no card.
* **Integer division (9 kernels).** New evidence, no card: (i) `test_host_gpu_ref` models the sequence read from the SASS of `mask-cuda` (cuobjdump-equivalent text from `VGPU_HOST_GPU_LOG=2`) and proves for EVERY divisor y (exhaustive with `VGPU_EXHAUSTIVE=1`; 1,049,573 divisors, every power of two and its neighbours, the extremes by default) that the estimate q1 satisfies `q1 * y <= 2^32 < (q1 + 2) * y` with the simulator's reciprocal and with one 1 ulp off either way (the card's `rcp.approx.ftz` is within 1 ulp of the exact value and the simulator's reciprocal within 0.5 ulp, so the two differ by at most one ulp); that makes `q = floor(x * q1 / 2^32)` at most the true quotient and at most 2 below it for every x, which the two corrections repair, so the result is `x / y` for any reciprocal in that range; 1,000,000 random and boundary (x, y) pairs run through the full model for each of the three reciprocals agree with `x / y`. 2 ulps off is NOT covered (the offset 0xffffffe is two units wide; 44 of 1.05 million divisors fail), which the test pins. `VGPU_EXHAUSTIVE=1 test_host_gpu_ref` (every divisor, 400 million random pairs for each of the three reciprocals) passes on server2 in 262 s. (ii) The simulator's execution of the real SASS (nvcc 13.0, sm_86) of `a / y` and `(int)a / (int)y` for all 2^32 nonzero divisors and 12 dividends (0, 1, 2, 3, INT_MAX, 2^31, 2^32-2, 2^32-1, 12345, 0x01234567, 0x9e3779b9, 0xdeadbeef), run in the simulator with `tools/probes/hostgpu_sfu_exhaustive.cu idiv` and `sidiv` (this round's new ops; the probe's "card" side is the program being run, here the simulator): 0 different in all 24 runs of 2^32 (3.5 to 3.9 minutes each); the same harness with its mutation switch (`HOSTGPU_SFU_MUTATE=1`, stride 257) reports 16,320 differences for `idiv` and 8,128 for `sqrtrn`. Together: the sim side is proven for the instruction sequence, and the card side is C semantics (an integer quotient has one value). **Not implemented:** the recognizer that tells a kernel's `I2F.RP`/`MUFU.RCP`/`F2I` chain is this sequence and not a hand-written `rcp.approx` with the same arithmetic around it. The loose match (the three instructions) would admit `cvt.rp.f32.u32; rcp.approx.ftz.f32; add 0x0ffffffe; cvt.rzi.ftz.u32.f32` written in inline PTX and consumed as a float-like value, which is NOT exact on the card; a sound recognizer must follow the chain through the Newton step (`IMAD.HI.U32` with the 64-bit addend pair) and both corrections, and its opt-in flag (`VGPU_HOST_GPU_IDIV`, by the pattern of `VGPU_HOST_GPU_DIV`) should wait for the card run below. That is a half-day of work with its own unit test on the SASS text above; it was not started because the brief for this round was "implement only if existing data proves it", and the proof above is new data and a model, not a card result.
* **Square root (4 kernels), reciprocal `1.0f / x`:** the card ops `sqrtrn` and `rcprn` are added to the probe; the simulator side was run here over all 2^32 patterns each: 0 different (17 s and 16 s). A recognizer of the `sqrt.rn` shape (`FSETP.GEU |x|, FLT_MIN`, `MUFU.RSQ`, FFMA chain, `CALL` to a slow path) is not written.
* **Bare `rsqrtf`, `expf`, `__frcp_rn` via `rcp.rn`, double division:** unchanged, refused (no idiom hides the approximation).

The exact card experiment, for the first session that has a card (one A10G or any sm_75+ card, 10 to 15 minutes, no AWS needed beyond the rig): `tools/probes/hostgpu_mufu_card_experiment.sh <out-dir>` builds the probe for the card's compute capability and runs, at stride 1 (all 2^32 patterns): E1 `rcp rsq sqrt ex2 lg2` (recorded, expected to differ), E2 `div` (12 dividends), E3 `sqrtrn rcprn idiv sidiv` (12 dividends for the two integer ops), E4 the harness's own mutation (each exact op must report differences when the card side is nudged by one ulp). It prints one `VERDICT <op> PASS|FAIL` per exact op and exits 0 only when all pass. Expected output (from the simulator runs of this round and sp15): every exact op `differ=0 tested=4294967296`; E1 lines `differ` in the hundreds of millions with `max_ulp` 1 or 2; `VERDICT div|sqrtrn|rcprn|idiv|sidiv|harness PASS`. Acceptance: `div` PASS lets the flag become the default (and the 13 kernels above are then in the set); `idiv` and `sidiv` PASS (with a sound recognizer written and `VERIFY=all` over mask, matrixT, backprop and md5hash showing `in_set_different=0`) admits the 9; `sqrtrn` PASS gates the sqrt recognizer. A FAIL line names the first differing pattern: the idiom is not exact on that card for that input and the set stays as it is. Do not enable any of them on a PASS of a smoke stride (the script says so).

Tests for this part: `test_host_gpu_ref` gains `integer_division_estimate_is_a_close_underestimate_for_every_divisor`, `integer_division_with_a_reciprocal_up_to_1_ulp_off_is_exact`, `integer_division_bounds_can_fail` and the `fsqrt_rn` / `frcp_rn` references in `reference_special_values`; the probe's new ops compile (nvcc 13.0, sm_86) and were run against the simulator at every stride used above.

### Mutants and results

Each mutant applied to a copy of the tree on server2, the four affected unit tests rebuilt and run, plus `e2e_host_gpu_stub` where the shim is involved, source restored (`/home/saqib/sp43-scratch/mutants.py`, 22 mutants):

| mutant | killed by |
|---|---|
| A1 no ski-rental account | `the_start_is_bought_when_as_much_was_spent_renting`, `the_first_launch_rule_pays_the_start_too`, e2e |
| A2 the start is never charged | four `test_host_gpu_logic` cases, e2e |
| A3 a single large launch ignores the start | four `test_host_gpu_logic` cases |
| A4 default start cost 0 | `the_start_cost_is_the_measured_one` and four more |
| A5 first-launch rule ignores the driver state | `the_first_launch_rule_pays_the_start_too` |
| A6 eager `begin_start()` at every exact-set launch | `e2e_host_gpu_stub` |
| A7 the decision always sees the driver as started | `e2e_host_gpu_stub` |
| A8 simulated time is not accounted | `e2e_host_gpu_stub` (step 8; no unit test reaches the runtime's `note_simulated`) |
| A9 the first-launch rule gets no account | `e2e_host_gpu_stub` |
| B1 every access changes state (no fast path) | `access_changes_state_is_exactly...`, three gate cases, the ratio test |
| B2 no access changes state | `access_changes_state_is_exactly...`, three gate cases |
| B3 a write to a shared chunk called harmless | `access_changes_state_is_exactly...`, `a_write_makes_the_chunk_the_simulators...` |
| B4 counts not reset by `sim_launch` | `residency_keeps_both_sides_coherent`, `residency_counts_match_a_recount...` |
| B5 `card_valid()` scans the chunks again | `residency_queries_do_not_grow_with_the_allocation` |
| B6 gate fast path removed | three gate cases, the ratio test |
| B7 gate treats a write as a read | `a_write_makes_the_chunk_the_simulators...` |
| B8 `card_launched_writing` does not keep the counts | `residency_counts_match_a_recount...` |
| C1 sqrt reference flushes subnormals | `reference_special_values` |
| C2 reciprocal reference is the approximate one | `reference_special_values` (after adding the subnormal-divisor case; first run: SURVIVED) |
| C3 integer-division offset one unit too big | the two integer-division proof tests |
| C4 integer-division model without the second correction | `integer_division_with_a_reciprocal_up_to_1_ulp_off_is_exact` |

Not covered by a mutant: the `before_access` wrapper in `host_gpu.cpp` that calls `gate_access` (a forwarding; the function it forwards to is tested, the wiring is exercised by the verify measurement above, not by a test) and the diagnostic lines of `VGPU_HOST_GPU_LOG`. The parsing of `VGPU_HOST_GPU_START_COST` has no mutant of its own: e2e steps 7 and 8 depend on it (a cost of 1000 s that was not read would start the driver in step 7).

### Default path unchanged

Unit suite on server2 (Release, g++ 13, `ctest -R '^test_'`, 376 tests): main d462bc2f 376 of 376 passed (built without the CUDA runtime shim: that configuration has no nvcc); this branch, rebuilt with nvcc 13.0 so that the shim and `e2e_host_gpu_stub` exist, 398 of 398 passed: the same 376 tests with the same result (no name missing, none failed) and 22 `test_cuda*`/`test_driver*`/`test_runtime*` tests that only exist with the shim. `e2e_host_gpu_stub` passes (with steps 7 and 8), `e2e_host_gpu` skips (no card). The host-GPU-off path executes none of the changed code, and with the mode on the program's output, exit codes, counters and digests were not touched (the mode refuses to start under counters, digests and the other diagnostics as before).

### Caveats

* Fix 1's "now" numbers use a stand-in driver with a 230 ms `cuInit`; the real start was measured at 0.23 s on the A10G in sp42. The rule was not run on a card: the effect on programs that the mode helps (a driver start after 0.25 s of simulation instead of at the first launch) is argued, not measured.
* Fix 2's number is a stand-in-driver run: the card "executes" nothing, so verify reports every kernel as different; the cost being measured is the simulator's side of the gate, which does not depend on what the card did.
* The proof for integer division covers 32-bit forms with one reciprocal estimate; 64-bit division (`I2F.U64.RP`, a `CALL` slow path) is a different sequence and was not looked at.
* `idiv` and `sidiv` runs in the simulator show the simulator's execution of the idiom is exact; they say nothing about the card, which is what the experiment script is for.

## Round sp47 (the blocks of one launch on several host threads, with the bits of running them in order: `VGPU_EXACT_THREADS`; branch `sp47-parallel-blocks`, base `origin/main` 27f09add)

Question: can a launch whose result must be that of `VGPU_THREADS=1` (blocks in order: the reference of `tools/verify`, of the counter checks and of the result cache) use several host cores, and what is that worth on Rodinia, PolyBench and HeCBench?

**What was there.** Blocks on several host threads are not new: `VGPU_THREADS=N` is the default (`Runner::run`, `launch_grid`; [host-parallelism.md](../host-parallelism.md)), with a pool, a width rule and lockstep for kernels that wait on one another. What it does not give is a proof: a kernel whose blocks share an atomic, or race, has one answer in order and one of several on threads, and nothing says which kind a kernel is. The held branch `sp4-threads` (f5798425) was about the cost of that mode, not its exactness: per-worker counter sums and an atomic failure flag (both on main since round 12, `worker_mt`) and the hand-out counter on a cache line of its own (measured as nothing in Round sp15 and dropped). It has nothing this round needed. [host-parallelism.md](../host-parallelism.md) H4 names why exactness was never attempted: whether blocks are independent "cannot be decided from the instruction stream in general (a store index is data)".

**What was built.** The decision is made while the launch runs instead of from the instruction stream. `VGPU_EXACT_THREADS=N` beside `VGPU_THREADS=1` (opt-in; unset, nothing changes) runs a SASS launch on `N` threads on trial under a shadow of device memory: one word per four bytes (untouched, read by one block, read by several, written by one block), moved by compare-and-swap before each access. A block that would read what another block wrote, or write what another read or wrote, stops the trial before making the access; a block's first write to a word keeps the old bytes. A trial with no stop is kept and is the run in order (each block read only the launch's start and its own stores, and wrote bytes no other block touched); any other is undone from the kept bytes and the launch runs in order as it always did. The argument, the table of every other channel by which a block could see the order (atomics, `%smid`, printf, device launches, textures, host and mapped memory, traces) and what each does, and the two things that are not the same (how memory is held after an undone trial, and time) are in [exact-threads.md](../../exact-threads.md). Code: `src/core/memory_spec.cpp` (shadow and undo log), the hooks in `src/core/memory.cpp` (35 methods; each is one null test when no trial is on), `Runner::run_exact` in `src/sass/exec.cpp`. Not statically decided: nothing is. A static rule only chooses the launches that are not tried at all (externs, clusters, cooperative, lockstep, textures, mapped memory).

Machines: server2 (Ryzen 9 9950X, 16 cores, runs CI runners and a database) for the ceiling, the tests, the A/B and the sanitizer run, programs on cpus 8-15 (eight cores, one thread each), `nice -n 19`, under the builder lock; server3 (Ryzen 7 PRO 8840U) for instruction counts and the mutation runs. Release builds without LTO (`-DVGPU_LTO=OFF`, as the other rounds on these hosts), `VGPU_GPU=nvidia/rtx3060`. Scripts and raw data: [data/sp47](../data/sp47).

### 1. The ceiling, measured before anything was built

`ceiling-log.patch` (not merged) logs each SASS launch: blocks, wall time inside the launch, the atomics it ran, and the static facts that keep a launch on one thread. One run of each program at `VGPU_THREADS=1` (`sp47_run.py ceiling`, `ceiling.json`). "Bound" is Amdahl's with the eligible share on N threads and no overhead; the host's load average was between 10 and 25 during this run (other sessions' builds), which moves the wall times and, less, the shares.

| corpus | program | wall s | in launches | in launches of 8 blocks or more | of those, none of the static rules | bound at 2 / 4 / 8 threads | kept on one thread by |
|---|---|---:|---:|---:|---:|---|---|
| polybench | 2DCONV | 0.61 | 43.4 % | 43.4 % | 43.4 % | 1.28 / 1.48 / 1.61 |  |
| polybench | 2MM | 115.42 | 34.8 % | 34.8 % | 34.8 % | 1.21 / 1.35 / 1.44 |  |
| polybench | 3DCONV | 0.50 | 74.1 % | 74.1 % | 74.1 % | 1.59 / 2.25 / 2.84 |  |
| polybench | 3MM | 1.31 | 63.7 % | 63.7 % | 63.7 % | 1.47 / 1.91 / 2.26 |  |
| polybench | ADI | 0.16 | 77.7 % | 0.0 % | 0.0 % | 1.00 / 1.00 / 1.00 |  |
| polybench | ATAX | 0.77 | 94.5 % | 94.5 % | 94.5 % | 1.90 / 3.44 / 5.79 |  |
| polybench | BICG | 0.19 | 65.2 % | 65.2 % | 65.2 % | 1.48 / 1.96 / 2.33 |  |
| polybench | CORR | 39.33 | 45.1 % | 45.1 % | 45.1 % | 1.29 / 1.51 / 1.65 |  |
| polybench | COVAR | 47.16 | 55.4 % | 55.4 % | 55.4 % | 1.38 / 1.71 / 1.94 |  |
| polybench | FDTD-2D | 76.54 | 97.4 % | 97.4 % | 97.4 % | 1.95 / 3.71 / 6.75 |  |
| polybench | GEMM | 0.44 | 70.2 % | 70.2 % | 70.2 % | 1.54 / 2.11 / 2.59 |  |
| polybench | GEMVER | 0.48 | 74.3 % | 74.3 % | 74.3 % | 1.59 / 2.26 / 2.86 |  |
| polybench | GESUMMV | 0.24 | 61.2 % | 61.2 % | 61.2 % | 1.44 / 1.85 / 2.15 |  |
| polybench | GRAMSCHM | 113.06 | 36.5 % | 36.2 % | 36.2 % | 1.22 / 1.37 / 1.46 |  |
| polybench | JACOBI1D | 0.73 | 90.6 % | 90.6 % | 90.6 % | 1.83 / 3.12 / 4.82 |  |
| polybench | JACOBI2D | 0.39 | 94.3 % | 94.3 % | 94.3 % | 1.89 / 3.41 / 5.71 |  |
| polybench | LU | 21.07 | 98.8 % | 98.7 % | 98.7 % | 1.97 / 3.85 / 7.32 |  |
| polybench | MVT | 0.86 | 84.9 % | 84.9 % | 84.9 % | 1.74 / 2.75 / 3.89 |  |
| polybench | SYR2K | 7.88 | 89.6 % | 89.6 % | 89.6 % | 1.81 / 3.05 / 4.62 |  |
| polybench | SYRK | 4.11 | 89.8 % | 89.8 % | 89.8 % | 1.81 / 3.06 / 4.66 |  |
| rodinia | backprop | 0.43 | 74.7 % | 74.7 % | 74.7 % | 1.60 / 2.27 / 2.89 |  |
| rodinia | gaussian | 3.03 | 99.7 % | 99.5 % | 99.5 % | 1.99 / 3.94 / 7.72 |  |
| rodinia | lud | 1.49 | 62.5 % | 62.2 % | 62.2 % | 1.45 / 1.87 / 2.19 |  |
| rodinia | needle | 0.52 | 86.1 % | 85.8 % | 85.8 % | 1.75 / 2.80 / 4.01 |  |
| rodinia | pathfinder | 0.91 | 33.7 % | 33.7 % | 33.7 % | 1.20 / 1.34 / 1.42 |  |
| rodinia | srad_v2 | 0.03 | 0.0 % | 0.0 % | 0.0 % | 1.00 / 1.00 / 1.00 |  |
| rodinia | particlefilter_float | 1.16 | 99.1 % | 99.1 % | 99.1 % | 1.98 / 3.89 / 7.50 |  |
| rodinia | sc_gpu | 0.53 | 91.9 % | 91.9 % | 91.9 % | 1.85 / 3.22 / 5.11 |  |
| rodinia | hotspot | 0.77 | 98.8 % | 98.8 % | 98.8 % | 1.98 / 3.86 / 7.36 |  |
| rodinia | bfs | 0.01 | 45.9 % | 45.9 % | 45.9 % | 1.30 / 1.53 / 1.67 |  |
| rodinia | nn | 0.01 | 6.2 % | 6.2 % | 6.2 % | 1.03 / 1.05 / 1.06 |  |
| rodinia | lavaMD | 1.30 | 98.9 % | 98.9 % | 98.9 % | 1.98 / 3.87 / 7.42 |  |
| hecbench | adam-cuda | 0.02 | 72.2 % | 72.2 % | 72.2 % | 1.57 / 2.18 / 2.72 |  |
| hecbench | aidw-cuda | 120.13 | 0.0 % | 0.0 % | 0.0 % | 1.00 / 1.00 / 1.00 | stopped at the time limit |
| hecbench | aligned-types-cuda | 146.69 | 47.3 % | 47.3 % | 47.3 % | 1.31 / 1.55 / 1.71 | stopped at the time limit |
| hecbench | aobench-cuda | 0.56 | 96.3 % | 96.3 % | 96.3 % | 1.93 / 3.61 / 6.37 |  |
| hecbench | aop-cuda | 1.34 | 96.5 % | 95.9 % | 11.5 % | 1.06 / 1.09 / 1.11 | atomics 1.13 s |
| hecbench | asmooth-cuda | 0.18 | 89.2 % | 89.2 % | 27.6 % | 1.16 / 1.26 / 1.32 | atomics 0.11 s |
| hecbench | asyncAllocation-cuda | 2.40 | 95.3 % | 95.3 % | 95.3 % | 1.91 / 3.51 / 6.02 |  |
| hecbench | atan2-cuda | 0.02 | 50.0 % | 50.0 % | 50.0 % | 1.33 / 1.60 / 1.78 |  |
| hecbench | atomicAggregate-cuda | 3.05 | 99.9 % | 99.9 % | 0.0 % | 1.00 / 1.00 / 1.00 | atomics 3.05 s |
| hecbench | atomicCost-cuda | 13.61 | 0.0 % | 0.0 % | 0.0 % | 1.00 / 1.00 / 1.00 | exit code -9 |
| hecbench | atomicIntrinsics-cuda | 0.15 | 88.4 % | 88.4 % | 0.0 % | 1.00 / 1.00 / 1.00 | atomics 0.13 s |
| hecbench | atomicPerf-cuda | 0.82 | 99.2 % | 99.2 % | 0.0 % | 1.00 / 1.00 / 1.00 | atomics 0.81 s |
| hecbench | atomicReduction-cuda | 120.03 | 99.4 % | 99.4 % | 0.0 % | 1.00 / 1.00 / 1.00 | atomics 119.34 s; stopped at the time limit |
| hecbench | attention-cuda | 0.02 | 31.8 % | 17.6 % | 1.2 % | 1.01 / 1.01 / 1.01 | atomics 0.00 s |
| hecbench | axpby-cuda | 0.29 | 86.8 % | 77.3 % | 77.3 % | 1.63 / 2.38 / 3.09 |  |
| hecbench | babelstream-cuda | 142.38 | 42.6 % | 42.6 % | 42.6 % | 1.27 / 1.47 / 1.59 | stopped at the time limit |
| hecbench | background-subtract-cuda | 0.04 | 0.0 % | 0.0 % | 0.0 % | 1.00 / 1.00 / 1.00 |  |
| hecbench | backprop-cuda | 0.01 | 0.0 % | 0.0 % | 0.0 % | 1.00 / 1.00 / 1.00 |  |
| hecbench | base64e-cuda | 0.37 | 0.0 % | 0.0 % | 0.0 % | 1.00 / 1.00 / 1.00 | exit code 2 |
| hecbench | bgmv-cuda | 0.00 | 0.0 % | 0.0 % | 0.0 % | 1.00 / 1.00 / 1.00 | exit code 1 |
| hecbench | bh-cuda | 0.01 | 20.1 % | 20.1 % | 20.1 % | 1.11 / 1.18 / 1.21 |  |
| hecbench | bilateral-cuda | 0.04 | 0.0 % | 0.0 % | 0.0 % | 1.00 / 1.00 / 1.00 |  |
| hecbench | binomial-cuda | 120.10 | 99.0 % | 99.0 % | 99.0 % | 1.98 / 3.88 / 7.45 | stopped at the time limit |
| hecbench | bitonic-sort-cuda | 0.21 | 92.2 % | 92.2 % | 92.2 % | 1.85 / 3.24 / 5.16 |  |
| hecbench | black-scholes-cuda | 16.71 | 83.7 % | 83.7 % | 83.7 % | 1.72 / 2.69 / 3.74 |  |
| hecbench | blockAccess-cuda | 0.01 | 53.3 % | 53.3 % | 53.3 % | 1.36 / 1.67 / 1.87 |  |
| hecbench | fpdc-cuda | 5.03 | 82.7 % | 82.7 % | 82.7 % | 1.70 / 2.63 / 3.61 |  |
| hecbench | perplexity-cuda | 0.30 | 92.9 % | 0.0 % | 0.0 % | 1.00 / 1.00 / 1.00 |  |
| hecbench | radixsort2-cuda | 1.66 | 98.8 % | 98.6 % | 0.0 % | 1.00 / 1.00 / 1.00 | atomics 0.41 s, polls (lockstep) 1.23 s |
| hecbench | s3d-cuda | 0.26 | 93.1 % | 93.1 % | 93.1 % | 1.87 / 3.32 / 5.41 |  |
| hecbench | shmembench-cuda | 3.50 | 99.8 % | 99.8 % | 99.8 % | 2.00 / 3.98 / 7.90 |  |
| hecbench | silu-cuda | 0.39 | 86.5 % | 86.5 % | 86.5 % | 1.76 / 2.85 / 4.11 |  |
| hecbench | sortKV-cuda | 0.13 | 85.3 % | 85.0 % | 0.0 % | 1.00 / 1.00 / 1.00 | atomics 0.03 s, polls (lockstep) 0.08 s |
| hecbench | zerocopy-cuda | 15.81 | 29.3 % | 29.3 % | 29.3 % | 1.17 / 1.28 / 1.34 |  |

| corpus (programs that ended by themselves with exit code 0) | wall s | in launches | 8 blocks or more | none of the static rules | bound at 2 / 4 / 8 threads |
|---|---:|---:|---:|---:|---|
| polybench (20) | 431.3 | 54.8 % | 54.7 % | 54.7 % | 1.38 / 1.70 / 1.92 |
| rodinia (12) | 10.2 | 85.6 % | 85.4 % | 85.4 % | 1.75 / 2.78 / 3.96 |
| hecbench (26) | 53.0 | 71.1 % | 70.5 % | 57.3 % | 1.40 / 1.75 / 2.00 |

Reading: launches of fewer than 8 blocks hardly exist in these corpora (the second and third percentage columns are equal almost everywhere; ADI and perplexity are the exceptions); what is not in launches is host code (PolyBench's CPU reference, file reading). The column "atomics" is launches that executed at least one atomic: the built version does not exclude them by rule (a block's atomics on words only it touches are exact), so the true eligible share is between the two columns; the A/B below shows what is kept in practice.

### 2. Speed and bits: `VGPU_THREADS=1` against `VGPU_EXACT_THREADS` 2, 4 and 8, and against plain `VGPU_THREADS=8`

`sp47_run.py ab`: five runs of each variant, interleaved run by run, median wall seconds of the whole program, cpus 8-15, the harness waiting while the load average was above 8. "bits" compares, in every run of the exact variants with the runs at `VGPU_THREADS=1`: the normalised program output and the files it wrote (as `tools/bench/real`), the exit code, `VGPU_COUNTERS_FILE` (every counter of every launch, in launch order) and `VGPU_MEMO_FINAL_HASH` (a hash of all live device memory at exit). `VGPU_THREADS=8` is the existing unproven mode: the ceiling of what a trial can reach.

| corpus | program | `VGPU_THREADS=1` s | exact 2 | exact 4 | exact 8 | `VGPU_THREADS=8` | trials kept / launches tried (8 threads) | bits |
|---|---|---:|---:|---:|---:|---:|---|---|
| polybench | 2DCONV | 0.60 | 0.65 (0.93x) | 0.52 (1.16x) | 0.45 (1.33x) | 0.38 (1.60x) | 1 / 1 | same |
| polybench | 3DCONV | 0.47 | 0.51 (0.93x) | 0.33 (1.45x) | 0.24 (1.94x) | 0.17 (2.74x) | 254 / 254 | same |
| polybench | 3MM | 1.30 | 1.36 (0.96x) | 0.91 (1.43x) | 0.69 (1.89x) | 0.57 (2.30x) | 3 / 3 | same |
| polybench | ADI | 0.16 | 0.22 (0.74x) | 0.15 (1.09x) | 0.15 (1.08x) | 0.10 (1.66x) | 2049 / 2049 | same |
| polybench | ATAX | 0.74 | 0.77 (0.97x) | 0.46 (1.63x) | 0.29 (2.58x) | 0.15 (5.06x) | 2 / 2 | same |
| polybench | BICG | 0.18 | 0.23 (0.77x) | 0.18 (1.02x) | 0.15 (1.24x) | 0.09 (2.00x) | 2 / 2 | same |
| polybench | GEMM | 0.43 | 0.40 (1.09x) | 0.28 (1.55x) | 0.21 (2.02x) | 0.17 (2.55x) | 1 / 1 | same |
| polybench | GEMVER | 0.41 | 0.49 (0.84x) | 0.34 (1.21x) | 0.26 (1.59x) | 0.17 (2.43x) | 3 / 3 | same |
| polybench | GESUMMV | 0.24 | 0.31 (0.78x) | 0.26 (0.91x) | 0.19 (1.26x) | 0.12 (2.03x) | 1 / 1 | same |
| polybench | JACOBI1D | 0.72 | 0.87 (0.83x) | 0.56 (1.29x) | 0.72 (1.00x) | 0.37 (1.95x) | 20000 / 20000 | same |
| polybench | JACOBI2D | 0.39 | 0.45 (0.87x) | 0.26 (1.48x) | 0.16 (2.37x) | 0.08 (5.14x) | 40 / 40 | same |
| polybench | LU | 20.86 | 18.80 (1.11x) | 10.67 (1.95x) | 6.38 (3.27x) | 3.21 (6.49x) | 2513 / 2513 | same |
| polybench | MVT | 0.82 | 0.83 (0.98x) | 0.57 (1.45x) | 0.35 (2.32x) | 0.22 (3.73x) | 2 / 2 | same |
| polybench | SYR2K | 8.02 | 7.10 (1.13x) | 4.26 (1.88x) | 2.77 (2.90x) | 1.93 (4.15x) | 1 / 1 | same |
| polybench | SYRK | 4.24 | 3.79 (1.12x) | 2.36 (1.80x) | 1.45 (2.93x) | 1.01 (4.21x) | 1 / 1 | same |
| rodinia | backprop | 0.41 | 0.45 (0.92x) | 0.29 (1.42x) | 0.21 (1.94x) | 0.14 (2.96x) | 2 / 2 | same |
| rodinia | gaussian | 3.08 | 2.08 (1.48x) | 1.12 (2.75x) | 0.65 (4.71x) | 0.48 (6.46x) | 511 / 511 | same |
| rodinia | lud | 1.41 | 1.24 (1.14x) | 0.93 (1.53x) | 0.79 (1.78x) | 0.78 (1.82x) | 124 / 124 | same |
| rodinia | needle | 0.23 | 0.19 (1.18x) | 0.14 (1.68x) | 0.10 (2.20x) | 0.06 (4.01x) | 253 / 253 | same |
| rodinia | pathfinder | 0.45 | 0.42 (1.08x) | 0.37 (1.23x) | 0.36 (1.26x) | 0.33 (1.36x) | 5 / 5 | same |
| rodinia | srad_v2 | 0.02 | 0.02 (0.90x) | 0.02 (0.91x) | 0.02 (0.90x) | 0.02 (0.91x) | 0 / 1 | same |
| rodinia | particlefilter_float | 1.17 | 0.77 (1.53x) | 0.40 (2.95x) | 0.28 (4.20x) | 0.68 (1.73x) | 27 / 36 | not comparable (5 different results in the runs at `VGPU_THREADS=1`) |
| rodinia | sc_gpu | 0.52 | 0.62 (0.83x) | 0.38 (1.37x) | 0.36 (1.43x) | 0.19 (2.77x) | 1432 / 1432 | same |
| rodinia | hotspot | 0.77 | 0.46 (1.65x) | 0.25 (3.08x) | 0.18 (4.28x) | 0.15 (5.01x) | 1000 / 1000 | same |
| rodinia | bfs | 0.01 | 0.01 (0.93x) | 0.01 (0.90x) | 0.01 (0.90x) | 0.00 (1.41x) | 5 / 26 | same |
| rodinia | nn | 0.01 | 0.01 (0.96x) | 0.01 (0.96x) | 0.01 (0.98x) | 0.01 (1.01x) | 1 / 1 | same |
| rodinia | lavaMD | 1.26 | 0.69 (1.82x) | 0.36 (3.53x) | 0.19 (6.53x) | 0.18 (7.06x) | 1 / 1 | same |
| hecbench | adam-cuda | 0.02 | 0.01 (1.51x) | 0.01 (2.21x) | 0.01 (3.04x) | 0.01 (2.94x) | 1 / 1 | same |
| hecbench | aobench-cuda | 0.50 | 0.26 (1.96x) | 0.13 (3.87x) | 0.07 (7.30x) | 0.07 (7.30x) | 1 / 1 | same |
| hecbench | aop-cuda | 1.46 | 0.92 (1.59x) | 0.48 (3.01x) | 0.31 (4.74x) | 0.25 (5.88x) | 201 / 201 | same |
| hecbench | asmooth-cuda | 0.18 | 0.17 (1.07x) | 0.15 (1.19x) | 0.14 (1.27x) | 0.04 (4.19x) | 2 / 3 | same |
| hecbench | asyncAllocation-cuda | 2.50 | 1.49 (1.67x) | 0.80 (3.10x) | 0.52 (4.82x) | 0.42 (5.97x) | 800 / 800 | same |
| hecbench | atan2-cuda | 0.01 | 0.01 (1.14x) | 0.01 (1.52x) | 0.01 (1.73x) | 0.01 (2.02x) | 3 / 3 | same |
| hecbench | atomicAggregate-cuda | 3.29 | 3.29 (1.00x) | 3.26 (1.01x) | 3.28 (1.00x) | 0.60 (5.46x) | 0 / 6 | same |
| hecbench | atomicIntrinsics-cuda | 0.15 | 0.15 (1.01x) | 0.15 (1.01x) | 0.15 (1.01x) | 0.03 (4.74x) | 0 / 4 | same |
| hecbench | atomicPerf-cuda | 0.86 | 0.56 (1.55x) | 0.38 (2.29x) | 0.29 (2.97x) | 0.20 (4.41x) | 9 / 18 | same |
| hecbench | attention-cuda | 0.01 | 0.02 (0.90x) | 0.02 (0.91x) | 0.02 (0.94x) | 0.01 (1.08x) | 1 / 2 | same |
| hecbench | axpby-cuda | 0.32 | 0.29 (1.08x) | 0.19 (1.65x) | 0.16 (1.91x) | 0.11 (2.94x) | 34 / 34 | same |
| hecbench | background-subtract-cuda | 0.00 | 0.00 (0.94x) | 0.00 (0.73x) | 0.00 (0.56x) | 0.00 (0.90x) | 0 / 0 | same |
| hecbench | backprop-cuda | 0.01 | 0.01 (0.88x) | 0.01 (1.13x) | 0.00 (1.37x) | 0.00 (1.57x) | 2 / 2 | same |
| hecbench | bh-cuda | 0.63 | 0.44 (1.43x) | 0.26 (2.45x) | 0.18 (3.47x) | 0.14 (4.47x) | 8 / 12 | same; `VGPU_THREADS=8` differs |
| hecbench | bilateral-cuda | 1.00 | 0.57 (1.73x) | 0.32 (3.08x) | 0.21 (4.85x) | 0.21 (4.84x) | 3 / 3 | same |
| hecbench | bitonic-sort-cuda | 0.22 | 0.17 (1.30x) | 0.10 (2.22x) | 0.07 (3.28x) | 0.07 (3.33x) | 136 / 136 | same |
| hecbench | black-scholes-cuda | 16.33 | 10.86 (1.50x) | 6.86 (2.38x) | 5.01 (3.26x) | 4.10 (3.98x) | 1 / 1 | same |
| hecbench | blockAccess-cuda | 0.01 | 0.01 (0.86x) | 0.01 (1.21x) | 0.01 (1.51x) | 0.00 (1.92x) | 4 / 4 | same |
| hecbench | fpdc-cuda | 5.67 | 5.86 (0.97x) | 3.01 (1.89x) | 3.96 (1.43x) | 2.71 (2.10x) | 1 / 1 | same |
| hecbench | perplexity-cuda | 0.30 | 0.21 (1.46x) | 0.12 (2.41x) | 0.12 (2.42x) | 0.10 (3.07x) | 1 / 1 | same |
| hecbench | radixsort2-cuda | 1.61 | 1.61 (1.00x) | 1.60 (1.00x) | 1.62 (1.00x) | 0.49 (3.30x) | 100 / 200 | same; `VGPU_THREADS=8` differs |
| hecbench | s3d-cuda | 0.25 | 0.35 (0.71x) | 0.21 (1.20x) | 0.14 (1.74x) | 0.08 (3.05x) | 54 / 54 | same |
| hecbench | shmembench-cuda | 3.60 | 1.83 (1.97x) | 0.91 (3.95x) | 0.50 (7.14x) | 0.50 (7.15x) | 1 / 1 | same |
| hecbench | silu-cuda | 0.38 | 0.30 (1.28x) | 0.18 (2.16x) | 0.12 (3.22x) | 0.10 (3.97x) | 50 / 50 | same |
| hecbench | sortKV-cuda | 0.13 | 0.13 (0.99x) | 0.13 (0.98x) | 0.13 (0.98x) | 0.05 (2.55x) | 8 / 16 | same; `VGPU_THREADS=8` differs |
| hecbench | zerocopy-cuda | 15.81 | 15.88 (1.00x) | 15.86 (1.00x) | 15.81 (1.00x) | 12.10 (1.31x) | 0 / 28 | same |

| geometric mean of the speed-ups | exact 2 | exact 4 | exact 8 | `VGPU_THREADS=8` |
|---|---:|---:|---:|---:|
| all (53) | 1.11x | 1.59x | 2.01x | 2.91x |
| hecbench (26) | 1.20x | 1.70x | 2.07x | 3.17x |
| polybench (15) | 0.93x | 1.39x | 1.85x | 2.91x |
| rodinia (12) | 1.16x | 1.66x | 2.06x | 2.43x |

Not in the table: 2MM, GRAMSCHM, FDTD-2D, CORR, COVAR (40 to 115 s each at one thread: five runs of five variants did not fit the time the host could be held; a one-run pass of three of them was queued and skipped: by then the host had under 8 GB of memory available, the floor set for it that night), and the HeCBench programs that hit the time limit or failed in the ceiling run at one thread (aidw, aligned-types, atomicReduction, babelstream, binomial, atomicCost, base64e, bgmv).

Reading:

* **Bits.** Every exact variant gave the output, files, counters and final memory hash of `VGPU_THREADS=1` in every run of every program, with one row that cannot be compared: `particlefilter_float` gives five different results in its five runs at `VGPU_THREADS=1` itself (the program is not deterministic; known from the Rodinia runs). Plain `VGPU_THREADS=8` differs from the run in order on `bh-cuda`, `radixsort2-cuda` and `sortKV-cuda`: those are the programs the proof is for, and the exact variants are the same on all three (their trials are stopped or their kernels poll and run in lockstep as at one thread). What differs there was looked at afterwards (`which_bits.py`, `which_bits.log`, three runs each at 8 threads): the program's output and the hash of all device memory at exit are the same as in order in every run; only the per-launch counters differ (instructions, thread instructions, global loads and their sectors and requests), and only in kernels that wait on other blocks: bh-cuda's `TreeBuildingKernel`, `SummarizationKernel` and `SortKernel` (6 of 17 launches; spin loops over `volatile` child, count and start arrays, and the `atomicCAS` lock of a tree cell), and CUB's one-sweep radix sort in the two Thrust programs (400 of 600 and 32 of 48 launches; its decoupled look-back reads other tiles' prefix descriptors until they are ready). How often such a loop goes round depends on when the other block's store arrives, on a card as here; so this is the programs' own timing dependence showing in exact counters, not a wrong result and not a fault of the simulator's threads.
* **Speed.** Geometric mean over 53 programs: 1.11x at 2 threads, 1.59x at 4, 2.01x at 8, against 2.91x for the unproven 8 threads. Summed wall time: 104.2 s at one thread, 89.1 / 61.5 / 50.5 s exact, 34.1 s at `VGPU_THREADS=8`. Best rows are at the unproven mode's speed (shmembench 7.2x, lavaMD 6.6x, aobench 7.1x at 8); worst are programs the mode cannot help (zerocopy: every trial stopped at a store to host-mapped memory, 0 of 28 kept, same time as one thread thanks to the hold-back; radixsort2 and sortKV: lockstep kernels and stopped trials).
* **Two threads do not pay on memory-bound kernels.** 21 of 53 programs are more than 3 % slower at 2 exact threads than at one (PolyBench 0.93x as a group). The shadow resets with every launch, so almost every access of a launch is a first touch: a compare-and-swap per lane, and for a store a copy of the old bytes. On the 33 programs of at least 0.2 s whose trials were all kept, 8 exact threads take 1.38x the time of 8 plain ones (geometric mean), and more on the loads-only PolyBench kernels (ATAX 0.29 s against 0.15 s, LU 6.4 s against 3.2 s). That factor is the price of the proof; it is why the mode is opt-in and why 2 threads are a loss where the kernel is mostly loads.
* **Trials kept.** Of the 53 programs, 11 had at least one trial stopped: the atomic programs (atomicAggregate 0 of 6, atomicIntrinsics 0 of 4, atomicPerf 9 of 18), bfs (5 of 26: its frontier flags are written by several blocks), bh-cuda (8 of 12), asmooth, attention, srad_v2, radixsort2 and sortKV (half), zerocopy (0 of 28). A stopped trial costs its work once and then the hold-back spaces the retries out.

### 3. What a run that does not ask for it pays

The hooks are a load and a branch in each memory-manager method. `default_cost.py`: base (origin/main with `ceiling-log.patch`) against this branch at `VGPU_THREADS=1` without the variable, cpu 6 of server3, alternating, median of 3: user instructions and cycles (`perf stat`), and `VGPU_COUNTERS_FILE` compared.

| program | user instructions, branch / base | cycles, branch / base | counters |
|---|---:|---:|---|
| polybench/ATAX | 1.0030 | 0.9907 | same |
| polybench/SYRK | 1.0021 | 1.0018 | same |
| polybench/3MM | 1.0033 | 1.0097 | same |
| polybench/JACOBI2D | 1.0012 | 1.0034 | same |
| polybench/MVT | 1.0030 | 1.0012 | same |
| polybench/LU | 1.0005 | 0.9984 | same |
| rodinia/hotspot | 1.0001 | 1.0082 | same |
| rodinia/gaussian | 0.9989 | 1.0459 | same |
| rodinia/lavaMD | 0.9990 | 0.9885 | same |
| rodinia/needle | 1.0002 | 1.0148 | same |
| hecbench/shmembench-cuda | 1.0000 | 1.0009 | same |
| hecbench/s3d-cuda | 1.0012 | 1.0045 | same |
| hecbench/aobench-cuda | 1.0000 | 0.9927 | same |
| hecbench/atomicPerf-cuda | 1.0003 | 0.9979 | same |

Instructions: 0.999 to 1.003 (the null tests). Cycles: within 1.5 % in both directions on 13 of 14, `gaussian` 1.046 (three runs each on a host other sessions were using; not explained, see Unresolved). Counters are the same in every run. Wall-clock A/B of the default afterwards (server3, load average under 4 enforced by the harness, cpu 6, `tools/bench/real/real.py`, 7 interleaved runs, medians; base is origin/main 7b90cee5 and the branch is the merge with it; `default_ab.log`):

| program | base s | branch s | branch / base |
|---|---:|---:|---:|
| gaussian | 3.60 | 3.65 | 1.014 |
| hotspot | 0.99 | 1.00 | 1.010 |
| lavaMD | 1.75 | 1.75 | 1.000 |
| SYRK | 5.37 | 5.39 | 1.004 |
| ATAX | 1.42 | 1.42 | 1.000 |
| shmembench | 4.82 | 4.86 | 1.008 |
| s3d | 0.38 | 0.39 | 1.00 (two digits) |
| gaussian, base against a copy of base | 3.61 | 3.61 | 1.000 |

The 1.046 in cycles for `gaussian` was mostly the shared host: on the quiet one it is 1.014 in wall time, with outputs the same. That 1.4 % comes with FEWER instructions than the base (0.999 above), so it is not the null tests being executed; it is where the memory manager's code landed (the layout effect this log has met before: equal instructions, different cycles). No change was made for it.

### 4. Tests, mutation evidence, sanitizers

* `test_exact_threads` (`tests/unit/test_exact_threads.cpp`, wired three times: default, `VGPU_TIER=0`, `VGPU_FASTPATH=0`): five hand-written sm_86 kernels (independent blocks that share a read; all blocks storing to the same words; a chain where block b reads what block b-1 stored; a fault in block 1 after later blocks could have stored; `%smid` on a grid larger than the device), each at 2, 3, 4 and 8 threads, three launches each, compared with the run in order word for word (128 K words), counter for counter (every `LaunchStats` word and the per-opcode vector) and error for error, and with the trial required to end the way the kernel dictates (kept for the first, stopped for the others); the hold-back schedule after a stop; no trial without `VGPU_THREADS=1`; every reading method of `MemoryManager` stopped at bytes another block wrote and every writing method at bytes another block read (26 methods; the two that decline must decline), the six things a trial may not do; a stopped trial undone to the byte across chunk ends, in never-written memory and at an allocation's tail that is no whole word; the shadow empty again after a trial.
* Mutations (`mutants.py`, each one wrong line, rebuilt and run in the three forms):

  | mutant | what is wrong | result |
  |---|---|---|
  | `write_conflict_ignored` | a write of a word another block read or wrote goes on | killed (all three forms) |
  | `read_of_written_ignored` | a read of a word another block wrote goes on | killed |
  | `readers_not_merged` | a second reader takes the word's mark instead of "several readers" (it could then write it) | killed |
  | `no_write_back` | a stopped trial does not put the kept bytes back | killed |
  | `nothing_kept` | zeros are kept instead of the old bytes | killed |
  | `kept_first_word_of_a_run_only` | only the first word of a run of newly owned words is kept | killed |
  | `shadow_not_cleared` | the shadow keeps its pages after a trial | killed |
  | `host_store_allowed` | a store outside the device's range is let through | killed |
  | `hook_store_batch`, `hook_load_run`, `hook_store_scalar`, `hook_load_scalar` | the method does not note its access | killed, each |
  | `window_not_declined` | `batch_window` hands out its pointer during a trial | killed |
  | `stopped_trial_kept` | a stopped trial is treated as kept | killed |
  | `blocks_not_run_again_from_zero` | the run in order after a stop starts where the trial's hand-out ended | killed |
  | `counters_not_summed` | only one worker's counters reach the launch | killed |
  | `smid_not_stopped` | `%smid` by host thread does not stop the trial | killed |
  | `no_hold_after_a_stop` | a stopped kernel is tried again at once | killed |
  | `failed_block_not_a_stop` | a block that throws does not stop the trial | **survived** the first run (every failing block of the test failed at an access the shadow stops anyway); the test `a_launch_whose_blocks_fail_without_touching_memory_fails_as_the_run_in_order_does` was added for it (a step budget that ends every block before its store); rerun after the merge of main 7b90cee5 (server3): **killed** in the three forms, and the unmutated test passes 11 of 11 in each |
  | `tried_without_blocks_in_order` | the variable is honoured without `VGPU_THREADS=1` | **survived**, and is equivalent on a host with more than one CPU: without `VGPU_THREADS=1` a launch of several blocks takes the several-thread path and never reaches the trial (the test `nothing_is_tried_without_the_variable_or_without_blocks_in_order` holds that from outside). On a one-CPU host the mutant would try launches the user did not ask to be in order; the results would still be those of one thread |

  20 mutants: 19 killed (one of them only by the test added for it), 1 survived and is equivalent on a host with more than one CPU; the unmutated test passes in the three forms (`mutants.log`, server3).

* ThreadSanitizer (server3, after the merge of main 7b90cee5; `-fsanitize=thread -g -O1`, Debug, the CI job's flags, `setarch -R`): `test_exact_threads` in its three forms 11 of 11 each, and `test_memory`, `test_memfast`, `test_memalloc`, `test_memrmw`, `test_memhot_sp8`, `test_load_batch16`, `test_memcopy_direct`, `test_memrmw_lockfree`, `test_memcount_forms`, `test_atom_rmw` all pass, with 0 ThreadSanitizer reports in every one (`tsan.log`). The rest of the suite under TSan is CI's.
* The same tests in a Release build of the merged tree, plus `test_host_gpu_logic` (54), `test_host_gpu_ref` (7), `test_host_gpu_track` (14) and `test_memo` (49): all pass (`merged_tests.log`).

### 5. Unresolved

* The trial's cost per access (1.38x at 8 threads, more on loads-only kernels) is first-touch compare-and-swaps. Not tried: taking two or four words with one wider compare-and-swap for a coalesced warp, an AVX2 test of a warp's 32 words before the scalar path, a shadow kept across the launches of one kernel. Until then 2 threads are a loss on memory-bound kernels.
* Default mode: `gaussian` is 1.4 % slower in wall time at 0.1 % fewer instructions (layout), the other six programs within 1 %. Not removed.
* Only the SASS engine. The PTX interpreter and the AMD engine run in order under the variable; clusters, cooperative launches, lockstep kernels, modules with any device runtime call (printf, malloc), textures, and devices with virtual-memory mappings or shared memory are never tried.
* The liveness rule (a worker looks at the stop flag after each round of a block's warps) has no test: mutant-free code is exercised, but no kernel in the test waits forever on another block's store.
* `config.md` / `vgpu` configuration key: the mode is an environment variable only.
* After a stopped trial, chunks the trial stored to stay materialized (bytes equal; host memory higher). No program in the corpora showed it as a problem; it is not measured.
* The five longest PolyBench programs and the eight HeCBench programs that do not finish at one thread within the limit have no A/B rows.
