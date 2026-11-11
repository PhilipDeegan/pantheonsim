# Host-GPU mode, relaxed: kernels whose result is the card's own

Status: experimental, opt-in (`VGPU_HOST_GPU=relaxed`), built without a card: the unit tests ran, the stand-in-driver test needs nvcc and
runs in CI, and nothing in this file has been measured on a GPU yet. The card measurements are *Round sp59* in
[rounds/11-rounds-sp36-sp41.md](rounds/11-rounds-sp36-sp41.md#round-sp59-relaxed-host-gpu-mode-card-measurements-to-follow).
Tracking issue: pantheonsim #587. Read [09-host-gpu-spike.md](09-host-gpu-spike.md) first: this mode is the same machinery with a wider
door.

## 1. Why

`VGPU_HOST_GPU=exact` sends a launch to the machine's real GPU only when the card computes the simulator's bits. Round sp42 measured what
that leaves behind: of the kernels the mode refused for their code, most held an approximate function (`MUFU`), and the rest a tensor-core
instruction, packed half arithmetic or an atomic. Round sp44 measured how far the card is from the simulator on the approximate functions:
4% to 25% of all float patterns differ, by one or two units in the last place.

Relaxed mode sends those kernels to the card as well. What the program then gets is the card's value, which is what the program gets on
real hardware, and which is close to the simulator's but not its bits. That is a different promise from exact mode's, so it is a different
mode, and exact mode is byte for byte what it was: the same set, defaults, decisions and summary text (a unit test holds a copy of the set).

## 2. The classes

Every instruction of a kernel is put in one class, once per kernel (`relaxed_class` in `include/vgpu/host_gpu.hpp`), and a kernel is
admitted by the classes it holds. It is still a whitelist: a mnemonic nobody named is `never`.

| class | instructions | default | switch |
|---|---|---|---|
| exact | today's exact set | admitted | (cannot be switched off) |
| mufu | every `MUFU` form, `RCP64H` and `RSQ64H` included | admitted | `VGPU_HOST_GPU_RELAXED_OFF=mufu` |
| half | `HADD2`, `HMUL2`, `HFMA2`, `HSETP2`, `HSET2`, `HMNMX2` | admitted | `=half` |
| float-red | `RED` / `REDG` on a float type (no value returned) | admitted | `=red` |
| call | `CALL` and `RET`, when every call is relative, to a label inside the kernel's own code section | admitted | `=call` |
| extra | `IDP`, `VABSDIFF`, `VABSDIFF4`, `VIADDMNMX`, `VIMNMX3`, `FMNMX3`, `F2IP`, `FSWZADD`, `REDUX`, `LDGSTS`, `LDGDEPBAR`, `LDSM`, `MOVM`, `FFMA2`, `FMUL2`, `FADD2`, integer `REDG` | admitted | `=extra` |
| tensor | `HMMA`, `IMMA`, `DMMA` | admitted | `=tensor` |
| atomic-int | `ATOM`, `ATOMG`, `ATOMS` on integers with `ADD`, `MIN`, `MAX`, `AND`, `OR`, `XOR`, `INC`, `DEC`; `RED` with `INC`, `DEC`, `POPC` | admitted | `=atomics` |
| atomic-float | `ATOM`, `ATOMG`, `ATOMS` on a float type (a value is returned) | refused | `VGPU_HOST_GPU_ATOMICS=1` |
| atomic-swap | any `CAS`, `CAST.SPIN`, `EXCH` | refused | `VGPU_HOST_GPU_ATOMICS=1` |
| never | `TEX`, `TLD`, `TLD4`, `TXD`, `SULD`, `SUST`; `QMMA`, `BMMA`, the sm_90 and sm_100 tensor and tensor-memory instructions; `JMP`, `NANOSLEEP`, `BPT`; a call that leaves the section; `ATOMS.ARRIVE`; an instruction with a field the decoder cannot name; anything else | refused | none |

Why each admitted class is safe to run on the card, as distinct from numerically equal: the card executes the module's own cubin, so any
instruction is "supported" there. What the mode must guarantee is that the card has everything the kernel reads. The shared checks do
that (section 3), and they are about memory and launch shape, not about opcodes. The classes above touch registers, shared memory and
global memory through the same addresses as `LDG` and `STG`. The `never` class holds what does not: textures and surfaces live in the
simulator's own tables, and the rest is unnamed because nobody has looked, not because it is known to be unsafe. The `extra` class is the
judgement call of this list: no measurement backs it yet, which is why it has a switch of its own.

A kernel of the exact set (the division idiom with `VGPU_HOST_GPU_DIV=1` included) is admitted whatever is switched off.

A refused kernel is named once on stderr, with the first instruction that refused it, so that a benchmark shows what is still missing:

```
host_gpu_refused: kernel=<name> class=never|off|atomics insn=<the instruction as nvdisasm prints it>
```

### The order-dependent classes

An integer atomic that returns the old value leaves the same memory whatever the order (the operations commute), but each thread's
returned value depends on the order the threads arrived in. A program that uses the value as an index (stream compaction, a work
queue) writes the same set of elements in another order on the card than in the simulator, and both are right. A float reduction adds in
arrival order, so its sum differs in the last places from run to run. Relaxed mode admits both by default; `VGPU_HOST_GPU_RELAXED_OFF=atomics,red`
keeps them in the simulator when a reproducible result matters more than speed.

A float atomic whose value is used, a compare-and-swap and an exchange are behind `VGPU_HOST_GPU_ATOMICS=1`. Section 2 of the spike found
the card disagreeing with itself on programs built on them (HeCBench `bh` tree building, `aop`): there is no single card answer, a verify
run can only report that the two sides differ, and a program's output can change from run to run. `tools/probes/hostgpu_atomic_repeat.cu`
measures which kinds a card repeats.

## 3. What is shared with exact mode, unchanged

Everything except the class scan: the launch-shape checks (cooperative and cluster launches, texture objects, graph conditionals, warp
size), arguments that point into mapped host memory, device memory outside the allocation table, modules with variables, external calls or
device-side launches, the size rule and its start cost, the first-launch rule, the bans, the 64 KiB residency protocol, the list of
diagnostics that switch the mode off, and `VGPU_HOST_GPU_MAX_MIB`.

Three places were checked for an assumption that silently depended on the exact set:

* **Calls.** The eligibility scan reads one code section. A kernel of the exact set could not call anything (the division idiom's slow
  path is checked instruction by instruction). In relaxed mode a `CALL` is admitted only when it is relative and its label is an address
  inside the section the scan read; a call through a register, to an absolute address, or into another section makes the kernel `never`,
  because it would run code nobody classed. Modules with external symbols were already refused. Recursion is the card's to run, on the
  card's own stack, as it is when the program runs natively.
* **Kernels that do not return.** `cuCtxSynchronize` has no deadline, a running kernel cannot be cancelled, and while it runs the card's
  memory cannot be read back, so there is no fallback. In exact mode the card computes the simulator's bits, so a kernel loops on the card
  exactly when it would loop in the simulator. In relaxed mode that argument is gone: a loop whose exit depends on a value the card
  computes differently, or a spin lock, is the card's to run. A correct program terminates on a real GPU, so the risk is the risk of running
  the program natively; what is new is that the simulator's own protection (`VGPU_MAX_STEPS`) is one of the diagnostics that switch the mode
  off. The guard is `VGPU_HOST_GPU_TIMEOUT` (seconds, default 600 in relaxed mode, none in exact mode): a launch that has been on the card
  longer makes the process say so and end with status 124. Spin locks and compare-and-swap loops are in the opt-in class as well.
* **The ski-rental account and the bans** count time, not opcodes: nothing to change. The simulator's time for a kernel that relaxed
  mode admits is learnt the same way (two simulated launches, or one long one).

Two bugs of exact mode's scan were found on the way:

* A kernel refused for its module (variables, external calls) was reported and counted as "not in the exact set" on every launch after its
  first, because the cached verdict kept no reason. The reason is cached now; such launches are counted as unsupported launch shapes every
  time. Fixed for both modes.
* `exact_mods_allowed` looks for the modifiers `F32`, `F64`, `F16` on a `RED`, but the decoder prints a float reduction's type as one
  modifier, `F32.FTZ.RN`. So exact mode admits float reductions on sm_75 to sm_89 (on sm_90 and later the mnemonic is `REDG`, which exact
  mode does not name). The fix is a separate commit, so that it can be judged apart from relaxed mode; relaxed mode's own classification
  reads the parts of joined modifiers and is not affected.

## 4. Verify by tolerance

`VGPU_HOST_GPU_VERIFY=1` (or `all`) runs each launch on the card and then in the simulator, as in exact mode. The simulator's result is the
one the program gets, so a difference does not carry into the next launch: verify measures each kernel alone. Without verify the program
gets the card's values and differences can compound over launches; that is the mode, and only an end-to-end comparison of the program's
output (the benchmark harness, section 6) says what it amounts to.

Exact mode compares bytes. Relaxed mode compares the same live ranges word by word (`include/vgpu/host_gpu_tol.hpp`), after skipping
64 KiB chunks that are equal. For each 4-byte word, aligned by address, whose bits differ:

1. as binary32: pass when `|card - sim| <= atol + rtol * |sim|`;
2. else as binary64, the aligned 8 bytes that hold the word, when they lie inside the range and the memory around the word does not
   look like binary32 data: the same rule;
3. else as two binary16 halves: pass when both halves pass.

A NaN matches any NaN under readings 1 and 2, and only the same bits under reading 3 (the low 16 bits of a float look like a half NaN
one time in 32). An infinity matches only the same infinity, and a word that is a binary32 infinity or NaN on one side is not tried as
halves. A word that passes no reading is outside tolerance. The defaults are
`VGPU_HOST_GPU_ATOL=1e-8` and `VGPU_HOST_GPU_RTOL=1e-5`; a value that is not a number zero or more turns the mode off with a message.
`VGPU_HOST_GPU_TOL_TYPE=f32|f64|f16` replaces the guess by one reading for every word (see the limits).

Each launch with a word outside tolerance is reported (the first three per kernel, all with `VGPU_HOST_GPU_LOG=1`): words compared,
different, outside, the largest absolute and relative difference, and the first address outside with both values in hex. At exit:

```
host_gpu_verify: in_set_identical=.. in_set_different=.. other_identical=.. other_different=..     (bytes, as in exact mode; "in set" = admitted)
host_gpu_relaxed: mode=relaxed launches_card=.. launches_sim=.. card_s=.. sim_s=.. words_compared=.. words_different=.. words_outside=.. words_subnormal=.. kernels_outside=.. max_abs=.. max_rel=..
host_gpu_refusals: refused_never=.. refused_atomics=.. refused_off=.. refused_exact=.. refused_safety=.. too_small=.. failed=..
host_gpu_outside: kernel=.. words_compared=.. words_different=.. words_outside=.. max_abs=.. max_rel=.. first=0x<address>:card=0x<word>:sim=0x<word>     (one per kernel with a word outside)
```

`host_gpu_relaxed:` and `host_gpu_refusals:` are printed by relaxed mode whenever the summary is (not under `VGPU_QUIET` alone), verify or
not: `launches_card`, `launches_sim`, `card_s` and `sim_s` are the coverage numbers of a plain run. Exact mode prints the two lines only
with `VGPU_HOST_GPU_LOG`, so that its summary is unchanged; there `refused_exact` counts the launches whose kernel relaxed mode would take.
`max_abs` and `max_rel` are taken under the reading that passed (binary32 for a word outside tolerance, when both sides are finite), and
`max_rel` only over differences above `atol`: below it the absolute term alone passes the word, and the ratio of two near-zero readings
(the low bits of a double, two small integers) says nothing.

### What changed from the first plan, and why

The plan was "else if the 8 bytes pass as binary64". Taken alone that rule makes verify blind to half of every float array: for a word at
an 8-byte boundary the next word, whatever it holds, reads as the high half of a double, and any change of the low half is at most 2^-20
of that double, which is under any rtol above 1e-6. So reading 2 is tried only where the simulator's bytes do not look like binary32
(`looks_like_f32`: in the 256 bytes around the word, three in four of the nonzero words at 8-byte boundaries have the exponent of a
magnitude between 2^-40 and 2^40; with fewer than four such words the word itself decides), and never for a word in the high half that is
a binary32 infinity or NaN on one side. A unit test changes each element of a float array by half its value and requires each to be
outside.

### Limits

Memory has no types, so the compare guesses, and these are the guesses that can be wrong:

* **Small integers pass.** Two integers below 2^23 read as two binary32 denormals, whose difference is far under `atol`. An index that is
  off by one is inside tolerance. Such words are counted (`words_subnormal`); a kernel with many is one to read by hand, or to run with
  `VGPU_HOST_GPU_ATOL=0`, which makes every denormal difference count.
* **Large integers and pointers fail**, as they should, but so does valid data that is merely in another order (section 2).
* **Where the data is not recognisably binary32** (sparse arrays, magnitudes outside 2^-40 to 2^40, structures) reading 2 can pass a wrong
  low word of an 8-byte pair.
* **Half precision is not judged by the guess.** Two halves in a word read, under reading 1, as one float whose exponent is the top
  bits of the high half. Halves below about 0.25 make a float below `atol`, so every difference between them passes; a difference in the
  low half is a change of the float's low mantissa bits and passes up to a quarter of its range. Memory that holds halves has to be
  compared with `VGPU_HOST_GPU_TOL_TYPE=f16` (every word as two halves, nothing else) and an rtol that takes one unit in the last place
  of a half, `VGPU_HOST_GPU_RTOL=2e-3`; float and integer memory of the same launch then shows as outside, so it is a way to look at one
  kernel, which is how the tensor probe uses it. `f32` and `f64` force the other readings the same way.
* **The guess has small leaks of its own** that the forced readings do not: a float whose low 16 bits change between `0x0000` and
  `0x8000` (0.4% of its value) passes as two halves, a zero of either sign and an equal one.
* **Within tolerance is not the same answer.** A difference of one unit in the last place can send a comparison the other way: a
  branch, a clamp, an iteration count, an index derived from a float. Verify reports what the memory looks like after one kernel, not
  what the program would have done next with the card's values.
* **Programs on compare-and-swap do not repeat on the card.** With `VGPU_HOST_GPU_ATOMICS=1` a difference from the simulator is not
  evidence of anything; compare the card with itself first.
* **Registers and shared memory are not compared**, only device memory in the allocation table, as in exact mode.

## 5. What a card has to say

Nothing here has run on a GPU. In order of how much depends on it:

1. `nvidia/tests/e2e/run_host_gpu.sh` and `tools/card-ci/run-card-ci.sh`: the expf kernel runs on the card in relaxed mode, verify finds
   no word outside the default tolerance, the output without verify differs only in the fields that hash expf results, exact unchanged.
2. `tools/probes/hostgpu_relaxed_card_probes.sh sfu <out>`: `sin`, `cos`, `tanh` over all 2^32 patterns (count, share, largest distance),
   beside sp44's five. The simulator side models `sin.approx` as a float product by 1/(2 pi) and `MUFU.SIN` in turns; a share above one
   half would point at that model, not the card.
3. `tools/probes/hostgpu_relaxed_card_probes.sh atomics <out>`: which kinds of atomic the card repeats run to run. Decides whether
   `atomic-int` and `float-red` deserve their default, and how bad the opt-in classes are.
4. `tools/probes/hostgpu_relaxed_card_probes.sh tensor <out> <build>`: `HMMA`, `IMMA`, `DMMA` kernels under verify, at the default rtol,
   at 2e-3, and at 2e-3 read as halves.
5. `tools/bench/hostgpu/run.py` (section 6) over the three corpora: speed-up, coverage, verify counts, end-to-end output.

## 6. The benchmark harness

`tools/bench/hostgpu/run.py` runs the prebuilt programs of `tools/bench/real` (PolyBench/GPU, Rodinia, the reduced HeCBench list; the same
layout, arguments and scratch directories) as sim, exact and relaxed, `--runs` times each, interleaved, and optionally once as relaxed with
`VGPU_HOST_GPU_VERIFY=all`. It writes JSON and a markdown table: wall time, speed-up over sim, launches on the card of all launches, seconds
on the card and in the simulator, launches kept in the simulator by reason, verify's counts, and the program's output against the sim
run's: identical, numeric (the same words, every number with a fraction or exponent within `atol + rtol * |sim|`; integers and hashes
must be equal), or different, with the largest numeric difference. A run past `--timeout` is killed (its own process group) and recorded.
`--dry-run` is a self-test that needs no card; it is the ctest `hostgpu_bench_selftest`.

```
python3 tools/bench/hostgpu/run.py --build build --corp /path/to/corp --gpu nvidia/a10g --runs 3 \
    --forms sim,exact,relaxed,verify --timeout 600 --json hostgpu.json --md hostgpu.md
```

## 7. Tests

| test | where it runs | what it pins |
|---|---|---|
| `test_host_gpu_logic` | every CI job | the mode names; exact's set against a frozen copy and every decoder mnemonic; each class; joined modifiers; calls; the switches and the off list |
| `test_host_gpu_tol` | every CI job | the rule, the three readings, NaN, infinity, denormals, signed zero, range ends, the float-array hole, parsing of the tolerances |
| `e2e_host_gpu_stub` | CI jobs with nvcc (a stand-in driver, no card) | relaxed reaches the card path for expf, rcp and division beside expf; exact does not; the lines for scripts; the atomics opt-in and the off list; unreadable values; verify against a card that computes nothing; the timeout |
| `hostgpu_bench_selftest` | every CI job | the harness: parsing, output classes, the table, a hang that is killed |
| `e2e_host_gpu`, `tools/card-ci/run-card-ci.sh` | a machine with a card only | section 5, item 1 |
