#!/usr/bin/env python3
"""tables.py out.json [sideA sideB ...]: per program medians and ratios (first side over each other side), geometric mean, the worst, spreads, JIT stats."""
import json, math, sys, statistics
d = json.load(open(sys.argv[1]))
key = "cold" if any("cold" in v for v in d.values()) and not any("warm" in v for v in d.values()) else "warm"
rows = []
for pid, v in d.items():
    if key not in v: continue
    w = v[key]
    if key == "cold":
        rows.append((pid, [r["wall"] for r in w["runs"]], [r.get("jit", {}) for r in w["runs"]]))
    else:
        rows.append((pid, w["med"], [w["samples"][i][0].get("jit", {}) for i in range(len(w["med"]))], w["spread"]))
if key == "cold":
    for pid, walls, jits in rows:
        print("%-34s" % pid, " ".join("%7.2f(c%d,%5.0fms)" % (w, j.get("compiles", 0), j.get("compile_ms", 0)) for w, j in zip(walls, jits)))
    sys.exit()
n = len(rows[0][1])
print("%-34s" % "program", " ".join("side%d med(spread)" % i for i in range(n)), "  ratios vs side0 (A/side)")
ratios = [[] for _ in range(n)]
for pid, meds, jits, sp in rows:
    line = "%-34s" % pid + " ".join("%7.2f(%3.0f%%)" % (m, 100 * s) for m, s in zip(meds, sp))
    for i in range(1, n):
        r = meds[0] / meds[i]; ratios[i].append(r); line += "  %.3f" % r
    ji = jits[-1]
    if ji.get("region_insts"): line += "  regions %.1fM insts (gen %.1fM) calls %.2fM" % (ji["region_insts"] / 1e6, ji["generated"] / 1e6, ji["region_calls"] / 1e6)
    print(line)
for i in range(1, n):
    r = ratios[i]
    print("side%d: geomean %.4f  min %.3f  max %.3f  slower than -2%%: %d" % (i, math.exp(sum(map(math.log, r)) / len(r)), min(r), max(r), sum(1 for x in r if x < 0.98)))
print("sum A %.1f  sums:" % sum(m[0] for _, m, *_ in rows), [round(sum(m[i] for _, m, *_ in rows), 1) for i in range(n)])
