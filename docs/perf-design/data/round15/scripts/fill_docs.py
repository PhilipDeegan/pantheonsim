#!/usr/bin/env python3
"""fill_docs.py: prints the generated tables of round15-jit.md (markdown) from the JSON in /tmp/sp33_res."""
import json, math, subprocess, sys
R = "/tmp/sp33_res/"
S = "/home/saqib/.cache/pantheonsim-sp33-jit-work/scripts/"
order = "polybench/SYRK,polybench/SYR2K,samples/convolutionSeparable,samples/mergeSort,samples/sortingNetworks,hecbench/fpdc-cuda,samples/interval,samples/histogram,hecbench/black-scholes-cuda,samples/jacobiCudaGraphs,polybench/LU,samples/simpleHyperQ,samples/alignedTypes,samples/radixSortThrust,polybench/FDTD-2D,hecbench/shmembench-cuda,samples/reduction,samples/transpose,samples/segmentationTreeThrust,samples/cudaTensorCoreGemm,hecbench/atomicAggregate-cuda,rodinia/gaussian"
def gm(xs): return math.exp(sum(map(math.log, xs)) / len(xs))
which = sys.argv[1]
if which == "warm":
    print(subprocess.run(["python3", S + "mdtable.py", R + "final_warm.json," + R + "final_warm2.json", order, R + "insts.json"], capture_output=True, text=True).stdout)
elif which == "x2":
    d = json.load(open(R + "x2_keep.json"))
    print("| program | tier s | base JIT s (A/B) | this branch (v2 build) s (A/B) | A/B base | **A/B v2** |")
    print("|---|---:|---:|---:|---:|---:|")
    rb, rv = [], []
    for pid, v in d.items():
        m = v["warm"]["med"]; rb.append(m[0] / m[1]); rv.append(m[0] / m[2])
        print("| %s | %.2f | %.2f | %.2f | %.3f | **%.3f** |" % (pid, m[0], m[1], m[2], m[0] / m[1], m[0] / m[2]))
    print("\ngeomean over these %d programs: base %.3f, this branch (v2 build) %.3f" % (len(rb), gm(rb), gm(rv)))
elif which == "cold":
    base = json.load(open(R + "final_cold.json")); v4 = json.load(open(R + "cold_v4.json"))
    warm = {}
    for f in ("final_warm.json", "final_warm2.json"): warm.update(json.load(open(R + f)))
    print("| program | tier s (warm table) | base JIT, cold: s (compiles, ms) | this branch, cold: s (compiles, ms) | cold / tier base | cold / tier this |")
    print("|---|---:|---|---|---:|---:|")
    tb, tv, cb, cv, nb, nv = [], [], 0, 0, 0, 0
    for pid in order.split(","):
        if pid not in base or pid not in v4 or pid not in warm: continue
        a = warm[pid]["warm"]["med"][0]
        rb = base[pid]["cold"]["runs"][0]; rv = v4[pid]["cold"]["runs"][0]
        jb, jv = rb["jit"], rv["jit"]
        cb += jb["compile_ms"]; cv += jv["compile_ms"]; nb += jb["compiles"]; nv += jv["compiles"]
        tb.append(rb["wall"] / a); tv.append(rv["wall"] / a)
        print("| %s | %.2f | %.2f (%d, %.0f) | %.2f (%d, %.0f) | %.2f | %.2f |" % (pid, a, rb["wall"], jb["compiles"], jb["compile_ms"], rv["wall"], jv["compiles"], jv["compile_ms"], rb["wall"] / a, rv["wall"] / a))
    print("\ncompiles: base %d (%.1f s), this branch %d (%.1f s); geomean of cold wall over tier wall: base %.3f, this branch %.3f" % (nb, cb / 1000, nv, cv / 1000, gm(tb), gm(tv)))
elif which == "cv":
    d = json.load(open("/tmp/sp33_res/cold_variants.json"))
    print("| program | default | `VGPU_JIT_RESIDENCE=0` | `VGPU_JIT_SPECIALIZE=0` | both |")
    print("|---|---|---|---|---|")
    for pid, v in d.items():
        r = v["cold"]["runs"]
        print("| %s | " % pid + " | ".join("%.1f s wall, %d compiles, %.0f ms" % (x["wall"], x["jit"]["compiles"], x["jit"]["compile_ms"]) for x in r) + " |")
elif which == "resid":
    d = json.load(open(R + "resid.json"))
    print("| program | tier s | JIT (rows in locals) s | JIT, `VGPU_JIT_RESIDENCE=0` (rows in memory) s | A/JIT | A/JIT residence off |")
    print("|---|---:|---:|---:|---:|---:|")
    r1, r2 = [], []
    for pid, v in d.items():
        m = v["warm"]["med"]; r1.append(m[0] / m[1]); r2.append(m[0] / m[2])
        print("| %s | %.2f | %.2f | %.2f | %.3f | %.3f |" % (pid, m[0], m[1], m[2], m[0] / m[1], m[0] / m[2]))
    print("\ngeomean over these %d programs: %.3f with residence, %.3f without" % (len(r1), gm(r1), gm(r2)))
