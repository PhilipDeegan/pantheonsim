Scripts of round 15 (sp33), written for server3 (paths `/tmp/sp33-jit/...` are what the rig used; edit them for another machine).

* `sp33.py`: the timing driver. `--sides build:tier|jit[:ENV=V,...],...` runs the sides interleaved per program (A,B then B,A, ...), medians of whole-process wall seconds,
  a first untimed run of every side (fills a JIT side's private cache), `--cold` for one run on an empty cache. Each run holds `/tmp/pantheonsim-builder.lock` for its own length only.
  It loads `real.py` (`tools/bench/real/real.py`, copy it beside it) for the program manifest and the output normalisation.
* `verify.py`: same bits: normalised output, `VGPU_COUNTERS_FILE` (pid removed) and `VGPU_KERNEL_DIGEST` of a reference run against variants, cold and warm cache, at the thread counts asked for.
* `tables.py`, `mdtable.py`: tables from the JSON `sp33.py` writes.
* `explore.sh`, `gate.sh` (the `e2e_sass_jit` shards one by one against prebuilt programs), `mutants.sh` (the JIT with `VGPU_JIT_MUTANT=<name>` against the tier on a prebuilt program),
  `unitmut.sh` (a unit test under each mutant), `perfprof.sh`, `ctime.sh` (compile-time experiments on a kept source), `wrap.sh` (wrapper scripts that make a real program a mutant target).
* `cp_bench.cpp`: the copy-and-patch stand-in of [copy-and-patch.md](../../../copy-and-patch.md).
