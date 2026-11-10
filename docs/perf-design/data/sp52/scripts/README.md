Scripts of round sp52, written for server3 (paths under `~/.cache/sp52` are what the rig used; `real.py` of `tools/bench/real` is copied beside them there).

* `count.py`: the exit reasons. Each program runs twice with the JIT and `VGPU_JIT_STATS=1` (cache fill, then warm); the warm run's region calls and its `jit: exit <reason>: n (diverged warp d, nothing run z)` lines go to a JSON file. `--env VGPU_JIT_LINK=1` for the linked side.
* `exittable.py`: the tables of the round from those JSON files.
* `sp33.py`: round 15's timing driver (interleaved sides, medians of whole-process wall seconds), with the JIT statistics off in timed runs unless `--stats`.
* `verify.py`: the same-bits driver (normalised output, `VGPU_COUNTERS_FILE`, `VGPU_KERNEL_DIGEST`, exit status against a reference side; cold and warm cache for JIT sides).
* `abtable.py`: the A/B table from `sp33.py`'s JSON.
* `mutants.sh`: the linking tests on the pristine build and under each mutant (needs `-DVGPU_JIT_MUTANTS=ON`).
