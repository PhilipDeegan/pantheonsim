#!/usr/bin/env python3
"""one.py --build B --corp C --prog polybench/SYR2K --cache DIR [--env K=V,K=V] [--preload sigprof.so --prof out.txt] [--cpu 6] [--runs N]
Round sp53: one program, run N times with the JIT on a given cache directory (the first run fills it), whole-process wall seconds printed per run.
With --preload the sampler is preloaded into the program (after `taskset env`, not into taskset) and writes its samples to --prof on the last run.
The caller holds /tmp/pantheonsim-builder.lock."""
import argparse, importlib.util, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("real", os.path.join(HERE, "real.py"))
real = importlib.util.module_from_spec(spec); spec.loader.exec_module(real)
ap = argparse.ArgumentParser()
ap.add_argument("--build", required=True); ap.add_argument("--corp", required=True); ap.add_argument("--prog", required=True)
ap.add_argument("--cache", required=True); ap.add_argument("--env", default=""); ap.add_argument("--preload", default=""); ap.add_argument("--prof", default="")
ap.add_argument("--cpu", default="6"); ap.add_argument("--runs", type=int, default=2); ap.add_argument("--stats", default="")
a = ap.parse_args()
byid = {p[0] + "/" + p[1]: p for p in real.manifest(a.corp, ["polybench", "rodinia", "hecbench"])}
corpus, name, cwd, argv, cp = byid[a.prog]
os.makedirs(a.cache, mode=0o700, exist_ok=True); os.chmod(a.cache, 0o700)
env = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_")}
env.update(VGPU_QUIET="1", VGPU_GPU="nvidia/rtx3060", LD_LIBRARY_PATH=a.build + "/shim", VGPU_THREADS="1", VGPU_JIT="1", VGPU_JIT_THRESHOLD="100000", VGPU_JIT_SYNC="1", VGPU_JIT_CACHE=a.cache)
env.update(dict(kv.split("=", 1) for kv in a.env.split(",") if kv))
for i in range(a.runs):
    e = dict(env); cmd = ["nice", "-n", "19", "taskset", "-c", a.cpu]
    last = i == a.runs - 1
    if last and a.stats:
        e.update(VGPU_JIT_STATS="1", VGPU_JIT_STATS_FILE=a.stats)
        if os.path.exists(a.stats): os.remove(a.stats)
    if last and a.preload:
        cmd += ["env", "LD_PRELOAD=" + a.preload, "SIGPROF_OUT=" + a.prof]
    t0 = time.monotonic()
    p = subprocess.run(cmd + argv, cwd=cwd, env=e, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("%s run %d rc %d wall %.3f" % (a.prog, i, p.returncode, time.monotonic() - t0), flush=True)
