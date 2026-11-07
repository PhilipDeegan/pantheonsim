# Copy-and-patch for tier 2? A study (sp33, 2026-11-07): not now; here is what would change that

Question (the brief): could copy-and-patch (Xu and Kjolstad, OOPSLA 2021, <https://arxiv.org/abs/2011.13127>: precompiled machine-code stencils with holes patched at run time, as in
CPython 3.13's JIT) replace the host-`g++` step of tier 2 -- no compiler at run time, microsecond compile latency, the JIT on by default -- given the repository's zero-dependency
rule and the `mmap(PROT_EXEC)` concern of [study 1](01-whole-kernel-translation.md#5-route-c-x86-64-template-jit)? Measured, not built; the implementation is not started because
@@REASON@@

## 1. What the compiler step costs today (measured)

@@COMPILE@@

## 2. What a stencil compile would cost (measured on a stand-in, not an implementation)

@@PATCH@@

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

@@QUALITY@@

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

@@RECOMMEND@@
