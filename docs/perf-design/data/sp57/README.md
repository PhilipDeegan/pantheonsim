Data of round sp57 (the section "Round sp57" of `../../rounds/11-rounds-sp36-sp41.md`): the JIT with its default settings against the tier, cold and warm.

* `t1-base.md`, `t1-base.json`: one thread, the base build (4464918a), all seven sides, 57 programs and fpdc.
* `t1-fix.md`, `t1-fix.json`: one thread, the build with the wait on the plain loop, default and `VGPU_JIT=1` cold and warm, 54 programs.
* `td-nobatch.md`, `td-nobatch.json`: the default thread count on that build before a thread added up its waited instructions in batches (six programs, the pass was stopped); `td-fix.md`, `td-fix.json`: after, 18 programs.
* `default-path-aa.txt`: no `VGPU_JIT`, the base build (`base`) against the changed one (`build`), interleaved: program, build, wall seconds.
* `failure-modes.txt`: `scripts/failure_modes.sh` on the changed build. `mutants.log`: `scripts/mutants.sh` on the final build. `tests.log`: the 24 ctest legs of the JIT's unit tests and the run-loop tests on the build before the batches.

The `.md` tables give the speed-up of each side (the default's median wall over the side's); the `.json` files hold, per program, the same-bits verdict, every run's wall by side (a: default; b: `VGPU_JIT=1`; c: with `VGPU_JIT_LINK=1`; d: with `VGPU_JIT_FUSE=1` too; second letter c cold, w warm) and the fill run's `[objects, bytes, compiles, compile ms, sections kept, declined]`.

Scripts (written for server3; `real.py` of `tools/bench/real` lies beside them there): `sp57.py` the driver, `table.py` its tables and the compact files, `failure_modes.sh`, `mutants.sh`.
