# Copy-and-patch for tier 2? A study (sp33, 2026-11-07): not now; here is what would change that

Question (the brief): could copy-and-patch (Xu and Kjolstad, OOPSLA 2021, <https://arxiv.org/abs/2011.13127>: precompiled machine-code stencils with holes patched at run time, as in
CPython 3.13's JIT) replace the host-`g++` step of tier 2 -- no compiler at run time, microsecond compile latency, the JIT on by default -- given the repository's zero-dependency
rule and the `mmap(PROT_EXEC)` concern of [study 1](01-whole-kernel-translation.md#5-route-c-x86-64-template-jit)? Measured, not built; the implementation is not started because
the evidence does not make it the clear next step; the numbers are in sections 1, 2, 4 and 6

## 1. What the compiler step costs today (measured)

Every number is this branch's JIT (graph regions, the generator of round 15) with g++ 13.3 on server3 (Ryzen 7 PRO 8840U), the study's protocol (threshold 100 000, synchronous compile, empty cache). Raw data `data/round15/final_cold.json`, `cold_v4.json`, `cold_variants.json`.

* **Per unit.** 67 units were compiled by the 22 programs' cold runs: **155.2 s in all, 2.3 s a unit on average** (base JIT, same programs: 37 units, 62.5 s, 1.7 s a unit). By program, from 0.6 s (simpleHyperQ, one unit) to 3 to 8 s for the programs with several units
  (mergeSort 4 units 8.1 s, transpose 8 units 7.6 s, alignedTypes 8 units 11.5 s, black-scholes 1 unit 7.1 s) and 76 s for segmentationTreeThrust (22 units); the table in [round15-jit.md](round15-jit.md) ("Cold cache") has every row.
* **What a unit costs, in detail.** SYRK's unit (4 regions, 207 ops, 148 KB of generated C++, 142,844 bytes of machine code, 21,450 instructions): 3.4 to 3.9 s wall at `-O2 -funroll-loops`, 345 MB resident; `-ftime-report` (wall column; its own total, 4.9 s, includes the report's overhead): parsing 1.63 s, deferred 0.04 s, optimisation and code
  generation 3.23 s. `#include "vgpu_jit_rt.h"` alone (the lane kernels, 1,671 lines, nothing generated) costs 0.42 s: a floor under every unit. Per op about 16 ms at `-O2` for these control-flow regions
  (the base JIT's regions of the same program: 1.1 s in all). On the kept source: one body per region instead of the two (whole warp, any group) 2.5 s; `-O1` with one body 1.5 s. Inside the JIT, cold (5 programs, `data/round15/cold_variants.json`): SYRK 4.6 s by default, 3.8 s without residence, 3.1 s with one body, 2.4 s with both.
  Also in the path: the compiler's identity (`--version`, `-dumpmachine`: about 7 ms, remembered in the cache), the cache lookup (9 to 17 ms), writing the source, the process spawn, `dlopen` and the checksum of the object. The compile is the whole of it.
* **What it costs a run.** Cold, the programs run 1.22x of the tier's wall time on geometric mean (21 programs; base JIT 1.14x), and exactly that is the compile: with the compile time taken out of the same runs they are 0.921x of the tier, against 0.905x warm. SYRK pays 1.75x, segmentationTreeThrust 3.2x.
  Round 14's default threshold (100 M instructions) keeps short programs from compiling at all, so nothing here is paid by the default settings on a run of a few seconds; it is what is paid by a long run's first minutes, or by every fresh CI runner that has no persistent cache directory.
* **What else the compiler step is.** A dependency at run time (`c++`, `g++` or `clang++` on `PATH`: the published Docker image has none by design), a writable and executable cache directory (a `noexec` mount stops `dlopen`), a child process with a time and address-space limit, and the reason the JIT is off by default.

## 2. What a stencil compile would cost (measured on a stand-in, not an implementation)

`cp_bench` (`data/round15/scripts/cp_bench.cpp`) does what the loader of a stencil JIT would do for N ops and nothing else: copies a stencil of 96 to 640 bytes (400 of them, 1 to 4 holes each, 370 bytes on average, the size of this JIT's compiled op bodies) into an anonymous read-write mapping, patches each hole with a 32-bit
value, `mprotect`s it read-execute. It is not an implementation (the bytes are random, nothing is executed, there is no relocation parsing), so it bounds the loader's cost and says nothing about the stencils'. Best of 9, cpu 9, server3:

| ops | stencil bytes | mmap | copy and patch | mprotect | total |
|---:|---:|---:|---:|---:|---:|
| 100 | 38 KB | 1.2 us | 11.1 us (110 ns an op) | 2.4 us | 14.7 us |
| 1,500 (the default `VGPU_JIT_MAX_OPS`) | 533 KB | 1.3 us | 155 us (104 ns an op) | 4.8 us | 162 us |
| 20,000 | 7.3 MB | 2.2 us | 2.38 ms (119 ns an op) | 32 us | 2.42 ms |

SYRK's 207 ops would be about 22 us, against 3.4 s for g++: 150,000 times less. The 67 units of the 22 programs: about 1.5 ms in all, against 155 s. Latency is not the question.

## 3. What would have to be built

A stencil JIT for this simulator is more than "memcpy the bodies":

* **Stencils.** The generated code is per-op AVX2 code over 32 lanes (the lane kernels of `src/simd/kernels.inc`): about 100 x86 instructions and 400 to 700 bytes per op
  (measured: the 207 ops of SYRK's unit are 142,844 bytes of text, half of it the second copy for "any group", about 690 bytes an op, 21,450 instructions). Each of the 29 forms the
  generator computes has operand shapes (register row, constant, register with a modifier, constant-bank word, uniform register, for up to three sources), guard shapes (none,
  @P, @!P, @UP, @!UP) and, for a store, a whole-warp and a partial-lane version. Holes can carry the register-file offsets, immediates, tables and branch targets (and are
  what copy-and-patch is good at), but not the *shapes*: those are different code. Roughly 150 to 300 stencils (50 to 150 KB of machine code), per ISA level (the kernels have a baseline and an
  AVX2 build, and the generated code follows the one the simulator runs).
* **A build step.** The stencils are C++ templates compiled when the simulator is built, then cut out of the object files: a tool (not in the tree; the zero-dependency rule
  means it is ours: an ELF reader of about 300 lines for the relocation types x86-64 gcc emits for such code) turns each function's bytes and relocations into a table in a generated header.
  Today's JIT has no build-time step of this kind; its generator is C++ text built at run time.
* **A loader.** `mmap` anonymous memory read-write, copy the stencils in the order of the ops, patch the holes, `mprotect` read-execute (or map the same memory twice, once
  writable and once executable), flush nothing (x86). One function per region with the same ABI (`Ctx`, `RegionFn`), so the executor side is unchanged.
* **Control flow.** The branches, the stretch checks and the exits are stencils with holes for the targets: forward and backward jumps inside the unit are patched rel32. That is what
  CPython's stencils do too.
* **Exceptions.** An escape that throws (a memory fault: `Fault`, `Error`) unwinds through the generated frame to `jit_run_region`'s `catch`. The compiler's object carries unwind
  tables (`-fexceptions`); copied code has none. Either the unwind info is registered for each unit (`__register_frame` and the per-unit FDE a patched copy needs -- libgcc
  specific, and a second thing to keep right per ISA) or an escape stops throwing through generated code: a trampoline catches, stores the exception and returns a flag the stencil
  after every escape tests (about 1 ns per escape, and a change in `tier_ops.inc`'s handlers' contract).
* **A second oracle.** Every op exists as an interpreter body, a tier handler and the generator's text (the differential tests hold each to the one before). A stencil is a fourth body unless
  the stencils are the generator's text compiled once with its operand kinds as parameters; the mutation switches (`VGPU_JIT_MUTANT=<form>`) would have to corrupt a stencil instead.


## 4. What stencil code would run like

The stencil approach gives up what the compiler does across ops. There are three kinds, and the first two can be measured here:

* **Rows kept in locals within a stretch ("residence")**: `VGPU_JIT_RESIDENCE=0` compiles the same regions with every op reading and writing the register file, as a stencil would. Warm, 3 runs, 7 programs: **1.229 with residence off against 1.202 with it** (geometric mean of tier/JIT; SYRK 1.125 against 1.109, LU 1.212 against 1.182, reduction 1.274 against 1.247, gaussian 1.077 against 1.018, simpleHyperQ 1.788
  against 1.728, transpose and mergeSort equal; full table in round15-jit.md). Residence does not pay in these programs, so losing it costs nothing measured.
* **Scheduling and inlining across ops** (the compiler sees a whole region, keeps predicates and addresses in registers, shares the `ex` computation): not measurable here without writing the stencils. A stencil body is the same lane kernels called per op, so an op costs its `*_k` calls plus the loads and stores of its operands; the compiled path measured 2.0 ns an op on a pure integer
  loop against the tier's 5.0 (study), 2.2 to 8.7 ns saved an op in programs. If stencils cost 1 to 1.5 ns more an op than compiled code, the 55 % of all instructions that run in generated code (9.96 G of 18.1 G; LU: 1,456 M of 1,777 M) would give back 10 to 15 s of the 22 programs' 664 s (1.5 to 2.3 %; LU 1.5 to 2.2 s of 25 s, 6 to 9 %), which is a third to a half of the JIT's gain on these programs (36 s: 1.057x
  corpus, 1.099x geometric mean). That is an estimate from a guess of the per-op difference, not a measurement.
* **Stencil-to-stencil control**: the generated code today runs a loop as one function with `goto`s; a stencil chain uses tail jumps (CPython's way), the same cost.

So: the residence experiment says the quality loss is not in the register-resident rows; the rest is unknown within a range that matters (0 to 2.3 % of the corpus time, up to 9 % of a program like LU).

## 5. Risks and blockers

* **`mmap(PROT_EXEC)`.** The tree has none today ([study 1](01-whole-kernel-translation.md): "no `PROT_EXEC` mappings in `src/`, `include/`, `nvidia/src/`"). Anonymous executable
  memory is refused by SELinux `execmem`/`execmod` policies, by some seccomp profiles and hardened containers, and on macOS needs `MAP_JIT` and an entitlement; W^X policies need the
  write-then-flip dance (two `mprotect` calls, a few microseconds a unit, measured below) or a double mapping from a `memfd` (which some container profiles refuse). Today's `dlopen` path
  fails differently (a `noexec` cache directory; no compiler), and the two failure sets only partly overlap: a rig with a read-only root and a `noexec` `/tmp` runs stencils and not
  `g++` output, one with SELinux enforcing runs neither or only the `dlopen`. A stencil JIT would have to keep the compiler path as a fallback to be no worse anywhere, so the
  compiler dependency would not go away, it would become optional.
* **Architecture.** x86-64 only (the generated code, its flags and the stencils). `ARCHITECTURE.md` assumes little-endian 64-bit hosts, not x86: on aarch64 the JIT is the tier today (no
  `-mavx2`, the kernels' baseline level), and would stay so until a second stencil set exists.
* **Code quality of the result.** See section 4: the stencils cannot keep rows in registers across ops, so the "residence" gain of the compiled path (which the compiler gets by seeing a whole
  region) is lost, and so is the compiler's scheduling across ops.
* **Debuggability.** No symbols for `perf` or a debugger inside the patched code (a `perf` map file can be written, nothing more); no source text to read when a region is wrong. The
  compiled path leaves `VGPU_JIT_KEEP=1` source and an `.so` with symbols.
* **Maintenance and testing** as above (a fourth body; mutants on stencils). The existing differential tests (`e2e_sass_jit_*`, `test_jit_resid`'s model) would apply unchanged to the
  same `RegionFn`, which is the one clear advantage of the design: the executor side and every test are shared.


## 6. Recommendation

**Do not build it now. Recommendation, with the numbers behind each part.**

1. *What it would buy.* The compile: 155 s for the 22 programs' cold runs (2.3 s a unit), which is the whole of the cold penalty (1.22x of the tier on geometric mean, 3.2x at worst; 0.92x without it). A stencil compile is 0.1 ms per 1,000 ops (cp_bench), five orders of magnitude less, no compiler on the machine, no
   cache directory, no first-run penalty: the JIT could be on by default at a threshold of a few hundred thousand instructions. It would also remove the dependency the published image lacks.
2. *What it would not buy.* Speed once compiled: the stencil code is at best the residence-off compiled code (1.229 against 1.202 measured on 7 programs) and probably a little slower (0 to 2.3 % of the corpus time, an estimate); the JIT's whole value on these programs is 1.057x on the corpus, 1.099x geometric mean, so the default-on JIT a stencil compile enables
   is worth about a tenth of the run time of programs that are long enough to matter, and nothing for programs that are dominated by memory, shared memory and atomics (a third of the 22). The big remaining gain (memory ops with exact counters in generated code: the study's extension (a), 1.05 to 1.28x) is independent of how the code is made, and needs its own design either way.
3. *What it would cost.* A build step that cuts stencils out of object files (an ELF reader of its own: the zero-dependency rule), 150 to 300 stencils for each of two ISA levels, a loader with W^X handling, a way for exceptions to leave generated frames without unwind tables (a trampoline per escape, about 1 ns each, or registering unwind info), x86-64 only, a fourth body per op to test and mutate,
   and the first `PROT_EXEC` mapping in the tree. Its failure set (`execmem` policies, hardened containers) overlaps the compiler path's only partly, so the compiler path stays as a fallback and the dependency becomes optional, not gone.
4. *What is cheaper and gets most of the cold gain.* `VGPU_JIT_RESIDENCE=0 VGPU_JIT_SPECIALIZE=0` (new knobs; residence existed): the compile 22 to 48 % lower (SYRK 4.6 s to 2.4 s, mergeSort 7.8 to 5.7 s) and the run time not worse. A persistent cache shared between CI jobs (the key is a function of the instruction words, the compiler and the flags, and the cache directory is portable): the compile is paid once per
   kernel, not once per runner. Both are a day's work, not a quarter's.
5. *What would flip it.* (a) A decision to make the JIT default-on for runs of seconds on hosts with no persistent cache and no compiler: only stencils do that. (b) If the memory-op extension lands and makes the JIT worth 1.3x, the cold compile (which grows with the generated code) stops being tolerable and stencils become the way to keep it. (c) Hosts where `g++` is absent
   and the JIT is wanted anyway. If one of these happens, start with the loader and 10 stencils for the integer forms and the branch, run `jit_random` and `e2e_sass_jit_shard_tier` against it (the `RegionFn` ABI is unchanged), and measure the one thing this study could not: the cost per op of stencil code against `VGPU_JIT_RESIDENCE=0` code.
