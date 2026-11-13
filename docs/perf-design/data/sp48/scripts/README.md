Scripts of round sp48, written for server3 (AMD Zen 4, IBS; paths under `~/.cache/sp48` are what the rig used).

* `prof.sh`: one IBS-op profile of one program (`ibs_op/cnt_ctl=1/` by micro-ops, then `cnt_ctl=0` by cycles, then `perf stat`), tier or JIT with a warm private cache. `NOLOCK=1` when the caller already holds `/tmp/pantheonsim-builder.lock`. The JIT only trusts a cache directory of mode 700 (the first batch of this round ran with 775, so its "jit" profiles were tier profiles: `ibs-tier.txt` calls them `tier-again`).
* `run_prof.py`: `prof.sh` over a list of programs of the corpus (`tools/bench/real/real.py`, copied beside it), printing `classify.py`'s split.
* `classify.py`: the classes of a profile (by function, and by host instruction inside ALU handlers, kernels and generated code). `reclass.sh` runs it again over kept `perf script` files; `ceiling.py` makes the tables of the round from that text; `lines.py` maps the samples of one function to source lines.
* `sp33.py`, `verify.py`: round 15's timing and same-bits drivers (`NOLOCK=1` added). `abtable.py`: the A/B table from `sp33.py`'s JSON.
* `mutants.sh`: the fused-run test on the pristine generator and under each mutant (needs a build with `-DVGPU_JIT_MUTANTS=ON`).

Data beside this directory: `ceiling-*.md` (the tables of the classified profiles, micro-ops and cycles; the raw classified profiles are not kept in the repository), `lines-run-loop.txt`, `jit-stats.txt`, `perf-stat.txt`, `warm_short.json`, `warm_long.json`, `ab-tables.md` (the A/B), `same-bits-*.log`, `mutants-fused.log`, `warm_more-unusable.log`. (The fused-runs patch as measured is not kept in the repository; round sp52 rebuilt it as `VGPU_JIT_FUSE`.)
