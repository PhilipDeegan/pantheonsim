#!/usr/bin/env python3
"""tables.py ab.json: the tables of Round sp55 from sp55_ab.py's output (markdown on stdout)."""
import json, math, sys

rows = json.load(open(sys.argv[1]))
gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")
X = ["x2", "x4", "x8"]
print("| corpus | program | `VGPU_THREADS=1` s (base) | base exact 2 / 4 / 8 | this round exact 2 / 4 / 8 | `VGPU_THREADS=8` | base twice | default, new / base | trials at 8: kept / stopped / again | bits |")
print("|---|---|---:|---|---|---:|---:|---:|---|---|")
for r in rows:
    t = r["a.t1"]
    sp = lambda k: "%.2fx" % (t / r[k])
    same = all(r[k + "_same"] for k in ["a.x2", "a.x4", "a.x8", "b.t1", "b.x2", "b.x4", "b.x8", "a.t1b"])
    bits = "same" if same else ("not comparable (`VGPU_THREADS=1` differs from run to run)" if not r["ref_one"] else
                                "DIFFERENT: " + ", ".join(k for k in ["a.x2", "a.x4", "a.x8", "b.t1", "b.x2", "b.x4", "b.x8", "a.t1b"] if not r[k + "_same"]))
    if same and not r["b.t8_same"]:
        bits += "; `VGPU_THREADS=8` differs"
    tr = r["trials"].get("b.x8", [0, 0, 0, 0])
    print("| %s | %s | %.2f | %s | %s | %s | %.3f | %.3f | %d / %d / %d | %s |" % (
        r["corpus"], r["name"], t, " / ".join(sp("a." + x) for x in X), " / ".join(sp("b." + x) for x in X), sp("b.t8"),
        r["a.t1b"] / t, r["b.t1"] / t, tr[0], tr[1], tr[2], bits))
print()
print("| geometric mean of the speed-ups | base exact 2 | 4 | 8 | this round exact 2 | 4 | 8 | `VGPU_THREADS=8` | base twice | default, new / base |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for name, sel in [("all (%d)" % len(rows), rows)] + [("%s (%d)" % (c, sum(1 for r in rows if r["corpus"] == c)), [r for r in rows if r["corpus"] == c])
                                                     for c in ("hecbench", "polybench", "rodinia")]:
    print("| %s | %s | %.2fx | %.3f | %.3f |" % (name, " | ".join("%.2fx" % gm([r["a.t1"] / r[b + x] for r in sel]) for b in ("a.", "b.") for x in X),
                                               gm([r["a.t1"] / r["b.t8"] for r in sel]), gm([r["a.t1b"] / r["a.t1"] for r in sel]),
                                               gm([r["b.t1"] / r["a.t1"] for r in sel])))
print()
tot = lambda k: sum(r[k] for r in rows)
print("Summed wall seconds: `VGPU_THREADS=1` %.1f; base exact %.1f / %.1f / %.1f; this round exact %.1f / %.1f / %.1f; `VGPU_THREADS=8` %.1f." % (
    tot("a.t1"), tot("a.x2"), tot("a.x4"), tot("a.x8"), tot("b.x2"), tot("b.x4"), tot("b.x8"), tot("b.t8")))
for b, n in (("a.", "base"), ("b.", "this round")):
    slow = [r["name"] for r in rows if r["a.t1"] / r[b + "x2"] < 0.97]
    print("%s: %d of %d programs more than 3 %% slower at 2 exact threads than at one%s" % (n, len(slow), len(rows), ": " + ", ".join(slow) if slow else ""))
big = [r for r in rows if r["a.t1"] >= 0.2 and r["trials"].get("b.x8", [0, 1])[1] == 0 and r["trials"].get("a.x8", [0, 1])[1] == 0]
print("Programs of at least 0.2 s whose trials were all kept (%d): 8 exact threads take %.2fx the time of 8 plain ones on the base, %.2fx this round." % (
    len(big), gm([r["a.x8"] / r["b.t8"] for r in big]), gm([r["b.x8"] / r["b.t8"] for r in big])))
long = [r for r in rows if r["a.t1"] >= 0.2]
aa = [r["a.t1b"] / r["a.t1"] for r in long]
dd = [r["b.t1"] / r["a.t1"] for r in long]
print("Programs of at least 0.2 s (%d): base twice %.3f (geometric mean; %.3f to %.3f); default mode, new / base %.3f (%.3f to %.3f)." % (
    len(long), gm(aa), min(aa), max(aa), gm(dd), min(dd), max(dd)))
for b, n in (("a.", "base"), ("b.", "this round")):
    print("%s, programs of at least 0.2 s: geometric mean %s at 2 / 4 / 8 exact threads; `VGPU_THREADS=8` %.2fx" % (
        n, " / ".join("%.2fx" % gm([r["a.t1"] / r[b + x] for r in long]) for x in X), gm([r["a.t1"] / r["b.t8"] for r in long])))
worse = [(r["name"], x, r["a." + x] / r["b." + x]) for r in long for x in X if r["a." + x] / r["b." + x] < 0.97]
print("Slower than the base by more than 3 % (programs of at least 0.2 s): " + ", ".join("%s %s %.2f" % (n, x, v) for n, x, v in worse))
better = sum(1 for r in long for x in X if r["a." + x] / r["b." + x] > 1.03)
print("Faster than the base by more than 3 %%: %d of %d program-thread counts; slower: %d." % (better, 3 * len(long), len(worse)))
