Scripts of round sp52, written for server3 (paths under `~/.cache/sp52` are what the rig used; `real.py` of `tools/bench/real` is copied beside them there).

* `count.py`: the exit reasons. Each program runs twice with the JIT and `VGPU_JIT_STATS=1` (cache fill, then warm); the warm run's region calls and its `jit: exit <reason>: n (diverged warp d, nothing run z)` lines go to a JSON file. `--env VGPU_JIT_LINK=1` for the linked side.
* `exittable.py`: the tables of the round from those JSON files.
* `sp33.py`: round 15's timing driver (interleaved sides, medians of whole-process wall seconds), with the JIT statistics off in timed runs unless `--stats`.
* `verify.py`: the same-bits driver (normalised output, `VGPU_COUNTERS_FILE`, `VGPU_KERNEL_DIGEST`, exit status against a reference side; cold and warm cache for JIT sides).
* `abtable.py`: the A/B table from `sp33.py`'s JSON.
* `mutants.sh`: the linking tests on the pristine build and under each mutant (needs `-DVGPU_JIT_MUTANTS=ON`).

Data beside this directory: `exits-unlinked.json`, `exits-linked.json` (the counts; `exit-tables.md`, `exit-tables-linked.md` are their tables), `ab-core.json`, `ab-more.json` (the A/B of 18 programs; `ab-tables.md`), `ab-syr2k.json`, `ab-syr2k-diag.json`, `ab-syr2k-diag2.json` (SYR2K again, and the three guesses that were not the cause), `cold.json` (first runs on an empty cache), `ab-fuse.json` (fused runs on top of linking), `same-bits-short.log`, `same-bits-long.log`, `same-bits-fuse.log`, `same-bits-tier-twice.log` (the tier against itself on the three programs that are not deterministic), `mutants.log`, `mutants-fused.log` (the fused runs' mutants on the combined build) and `tests.log` (the unit and run-loop tests of the measured build).
