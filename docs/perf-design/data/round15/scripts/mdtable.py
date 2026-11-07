#!/usr/bin/env python3
"""mdtable.py out1.json[,out2.json...] [program,program,...]: markdown tables. A file's sides are (tier, tier again, JIT) or (tier, JIT)."""
import json, math, sys
d = {}
for f in sys.argv[1].split(","):
    d.update(json.load(open(f)))
order = sys.argv[2].split(",") if len(sys.argv) > 2 else list(d)
insts = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {}
rows = []
for pid in order:
    if pid not in d or "warm" not in d[pid]: continue
    w = d[pid]["warm"]; m = w["med"]; sp = w["spread"]
    ji = len(m) - 1
    j = w["samples"][ji][0].get("jit", {})
    rows.append(dict(pid=pid, A=m[0], Ap=m[1] if len(m) == 3 else None, J=m[ji], sA=sp[0], sJ=sp[ji], jit=j, hashes=w["hashes"]))
def gm(xs): return math.exp(sum(map(math.log, xs)) / len(xs))
print("| program | tier A s (spread) | tier again A' s | JIT s (spread) | **A/JIT** | A/A' | warp instructions M | ran in regions % | computed by generated code % | region calls M |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
rj, raa = [], []
for r in rows:
    x = r["A"] / r["J"]; rj.append(x)
    aa = r["A"] / r["Ap"] if r["Ap"] else None
    if aa: raa.append(aa)
    j = r["jit"]
    tot = insts.get(r["pid"], 0)
    print("| %s | %.2f (%d%%) | %s | %.2f (%d%%) | **%.3f** | %s | %s | %s | %s | %s |" % (r["pid"], r["A"], round(100 * r["sA"]), ("%.2f" % r["Ap"]) if r["Ap"] else "-", r["J"], round(100 * r["sJ"]), x,
          ("%.3f" % aa) if aa else "-", ("%.0f" % (tot / 1e6)) if tot else "-", ("%.0f" % (100 * j["region_insts"] / tot)) if tot and j.get("region_insts") else "0", ("%.0f" % (100 * j["generated"] / tot)) if tot and j.get("generated") else "0",
          ("%.2f" % (j["region_calls"] / 1e6)) if j.get("region_calls") else "0"))
print()
worst = min(range(len(rj)), key=lambda i: rj[i]); best = max(range(len(rj)), key=lambda i: rj[i])
print("programs %d; geomean A/JIT **%.4f**; worst %.3f (%s); best %.3f (%s); slower than -2%%: %s; sum A %.0f s, sum JIT %.0f s (%.3fx)" % (
    len(rows), gm(rj), rj[worst], rows[worst]["pid"], rj[best], rows[best]["pid"], ", ".join(rows[i]["pid"] for i, x in enumerate(rj) if x < 0.98) or "none",
    sum(r["A"] for r in rows), sum(r["J"] for r in rows), sum(r["A"] for r in rows) / sum(r["J"] for r in rows)))
if raa: print("A/A' over %d programs: %.3f to %.3f (geomean %.4f)" % (len(raa), min(raa), max(raa), gm(raa)))
bad = [r["pid"] for r in rows if len(set(tuple(h) for h in r["hashes"])) != 1]
print("output hashes: " + ("equal across sides in every program" if not bad else "DIFFER in " + ", ".join(bad)))
