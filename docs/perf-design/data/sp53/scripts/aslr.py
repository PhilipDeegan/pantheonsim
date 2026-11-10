#!/usr/bin/env python3
"""aslr.py --build B --corp C --prog polybench/SYR2K --sides u=,l=VGPU_JIT_LINK=1 --pads 0,16,32 --reps 1 --out f.json [--aslr] [--cache-root D]
Round sp53: is a time difference between two settings a matter of where things lie in memory? Each side (name=ENV=V;ENV=V) is run on its own warm cache
with address-space randomisation off (`setarch -R`; --aslr leaves it on) and the stack moved by an environment variable of `pad` bytes (VGPU_PAD, which
nothing reads), sides and pads interleaved, `reps` times; whole-process wall seconds per run. The caller holds /tmp/pantheonsim-builder.lock."""
import argparse, importlib.util, json, os, platform, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("real", os.path.join(HERE, "real.py"))
real = importlib.util.module_from_spec(spec); spec.loader.exec_module(real)
ap = argparse.ArgumentParser()
ap.add_argument("--build", required=True); ap.add_argument("--corp", required=True); ap.add_argument("--prog", required=True); ap.add_argument("--sides", required=True)
ap.add_argument("--pads", default="0"); ap.add_argument("--reps", type=int, default=1); ap.add_argument("--out", required=True); ap.add_argument("--aslr", action="store_true")
ap.add_argument("--cache-root", default=os.path.expanduser("~/.cache/sp53/work")); ap.add_argument("--cpu", default="6")
a = ap.parse_args()
byid = {p[0] + "/" + p[1]: p for p in real.manifest(a.corp, ["polybench", "rodinia", "hecbench"])}
corpus, name, cwd, argv, cp = byid[a.prog]
sides = []
for s in a.sides.split(","):
    n, _, e = s.partition("="); sides.append((n, dict(kv.split("=", 1) for kv in e.split(";") if kv)))
def run(side, env_extra, pad, timed=True):
    cache = os.path.join(a.cache_root, "aslr-%s-%s" % (a.prog.replace("/", "_"), side))
    os.makedirs(cache, mode=0o700, exist_ok=True); os.chmod(cache, 0o700)
    env = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_")}
    env.update(VGPU_QUIET="1", VGPU_GPU="nvidia/rtx3060", LD_LIBRARY_PATH=a.build + "/shim", VGPU_THREADS="1", VGPU_JIT="1", VGPU_JIT_THRESHOLD="100000", VGPU_JIT_SYNC="1", VGPU_JIT_CACHE=cache)
    env.update(env_extra)
    if pad: env["VGPU_PAD"] = "x" * pad
    cmd = ["nice", "-n", "19", "taskset", "-c", a.cpu] + ([] if a.aslr else ["setarch", platform.machine(), "-R"]) + argv
    t0 = time.monotonic()
    p = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dict(side=side, pad=pad, rc=p.returncode, wall=round(time.monotonic() - t0, 3))
res = []
for n, e in sides: run(n, e, 0)   # the cache fill
pads = [int(x) for x in a.pads.split(",")]
for r in range(a.reps):
    for pad in pads:
        for n, e in (sides if r % 2 == 0 else sides[::-1]):
            x = run(n, e, pad); res.append(x); print(x, flush=True)
            json.dump(dict(prog=a.prog, aslr=a.aslr, runs=res), open(a.out, "w"), indent=0)
