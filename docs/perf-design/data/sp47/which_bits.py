#!/usr/bin/env python3
"""Round sp47: for the programs where plain VGPU_THREADS=8 differs from VGPU_THREADS=1, WHAT differs: the program's output, the final
device memory, or the per-launch counters (and then which kernels and which counter fields).
  which_bits.py --build B --corp C --real <tools/bench/real> --cpus 0-7"""
import argparse, hashlib, json, os, re, subprocess, sys, tempfile, shutil

ap = argparse.ArgumentParser()
for k in ("build", "corp", "real"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--cpus", default="0-7")
ap.add_argument("--programs", default="hecbench/bh-cuda,hecbench/radixsort2-cuda,hecbench/sortKV-cuda")
a = ap.parse_args()
sys.path.insert(0, a.real)
import real
man = {e[0] + "/" + e[1]: e for e in real.manifest(a.corp, ["polybench", "rodinia", "hecbench"])}


def run(e, threads):
    corpus, name, cwd, argv, cp = e
    d = tempfile.mkdtemp(prefix="sp47wb-")
    env = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_")}
    env.update(VGPU_QUIET="1", VGPU_GPU="nvidia/rtx3060", LD_LIBRARY_PATH=os.path.join(a.build, "shim"), VGPU_THREADS=str(threads),
               VGPU_COUNTERS_FILE=os.path.join(d, "c"), VGPU_MEMO_FINAL_HASH=os.path.join(d, "f"))
    p = subprocess.run(["taskset", "-c", a.cpus] + argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
    out, _ = real.normalise(p.stdout.decode("utf-8", "replace"), b"")
    launches = [json.loads(l) for l in open(os.path.join(d, "c"))] if os.path.exists(os.path.join(d, "c")) else []
    final = open(os.path.join(d, "f")).read() if os.path.exists(os.path.join(d, "f")) else ""
    shutil.rmtree(d, ignore_errors=True)
    return out, final, launches


for name in a.programs.split(","):
    ref = run(man[name], 1)
    again = run(man[name], 1)
    print("%s: VGPU_THREADS=1 twice: output %s, final memory %s, counters %s" % (
        name, ref[0] == again[0], ref[1] == again[1],
        [{k: v for k, v in x.items() if k != "pid"} for x in ref[2]] == [{k: v for k, v in x.items() if k != "pid"} for x in again[2]]))
    for rep in range(3):
        got = run(man[name], 8)
        fields, kernels, n = {}, {}, 0
        if len(got[2]) != len(ref[2]):
            print("  run %d: %d launches against %d" % (rep, len(got[2]), len(ref[2])))
        for x, y in zip(ref[2], got[2]):
            diff = [k for k in x if k not in ("pid", "launch", "seq") and x.get(k) != y.get(k)]
            if diff:
                n += 1
                kernels[str(x.get("kernel"))[:60]] = kernels.get(str(x.get("kernel"))[:60], 0) + 1
                for k in diff:
                    fields[k] = fields.get(k, 0) + 1
        print("  VGPU_THREADS=8 run %d: output %s, final memory %s, launches whose counters differ %d of %d" % (
            rep, "same" if got[0] == ref[0] else "DIFFERS", "same" if got[1] == ref[1] else "DIFFERS", n, len(ref[2])))
        if n:
            print("     kernels: %s" % sorted(kernels.items(), key=lambda kv: -kv[1])[:4])
            print("     fields: %s" % sorted(fields.items(), key=lambda kv: -kv[1])[:8])
