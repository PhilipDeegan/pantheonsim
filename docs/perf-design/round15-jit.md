# Round 15 (sp33): tier 2 (the JIT) with control flow in generated code

Branch `sp33-jit` (from `origin/main` 7d8d98a4), 2026-11-07. Design: [06-tier2-jit.md](06-tier2-jit.md); what this is measured against: the study of 2026-11-06
(`~/.cache/pantheonsim-briefs/jit-study-2026-11-06.md`, called **the study** below) and today's JIT (rounds 11 to 14, called **base** below). The copy-and-patch question
is its own document: [copy-and-patch.md](copy-and-patch.md). Raw data: `data/round15/`.

@@VERDICT@@

## Why the generated code covered half of what ran in regions (the study's open question)

The study's table 1 has two columns that look like a contradiction: SYRK 81.1 % of its instructions "ran in regions" and 45.5 % were "computed by generated code" (SYR2K
90.4 and 56.8). It is not a contradiction and nothing was lost. "Ran in regions" counts the **escapes**: an LDG or STG inside a region is a call of the tier's own handler, and it
is counted as a region instruction. SYRK is 51.8 % loads and stores; its regions held 67 M escaped loads (the JIT statistics: 157.6 M instructions ran in 33.6 M calls, 88.4 M of them
computed, so 69 M were escapes) and the remaining ~36 % were branches, S2R and uniform ops, which ended the regions. The generated coverage is bounded by the opcode set
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
* `Ctx` ABI 3 (`kAbiVersion`, `kGeneratorVersion` 3): `up` and `ur` are writable, new fields `divergent`, `gate`, `budget`, `extra`, `gen_run`, `sreg`, `max_steps`, `tctx`, `ops`, `last`, `exit_idx`, `exit_ps`.

@@RESULTS@@

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

`nvidia/tests/e2e/run_sass_jit.sh` can now run on a rig without nvcc with programs built beforehand (`VGPU_TIER_WORK`, `jit_random.<seed>.<arch>`).

@@MUTANTS@@

