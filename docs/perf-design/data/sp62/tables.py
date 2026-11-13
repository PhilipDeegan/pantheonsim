#!/usr/bin/env python3
"""Round sp62 tables from ab_reduced.json (sp62_ab.py) and tune.json (tune.py): tables.py ab_reduced.json tune.json"""
import json, sys
rows = json.load(open(sys.argv[1]))
print("| program | `VGPU_THREADS=1` s (base) | base exact 2 / 4 / 8 | this round exact 2 / 4 / 8 | new / base speed at 2 / 4 / 8 | `VGPU_THREADS=8` | base twice | default, new / base | bits |")
print("|---|---:|---|---|---|---:|---:|---:|---|")
for r in rows:
    t1 = r["a.t1"]
    sp = lambda k: t1 / r[k]
    same = all(r[k + "_same"] for k in ("a.x2", "a.x4", "a.x8", "b.t1", "b.x2", "b.x4", "b.x8", "a.t1b"))
    print("| %s | %.2f | %s | %s | %s | %.2fx | %.3f | %.3f | %s |" % (
        r["name"], t1, " / ".join("%.2fx" % sp("a.x%d" % n) for n in (2, 4, 8)), " / ".join("%.2fx" % sp("b.x%d" % n) for n in (2, 4, 8)),
        " / ".join("%.3f" % (r["a.x%d" % n] / r["b.x%d" % n]) for n in (2, 4, 8)), sp("b.t8"), t1 / r["a.t1b"], t1 / r["b.t1"],
        "same" if same else "DIFFERS"))
if len(sys.argv) > 2:
    S = ["0,0", "0,1", "2,1", "4,0", "4,1", "8,1"]
    print("\n| program, threads | base s | " + " | ".join("new %s" % s for s in S) + " |")
    print("|---|---:|" + "---:|" * len(S))
    for r in json.load(open(sys.argv[2])):
        for n in ("4", "8"):
            b = r["base.x" + n]
            print("| %s x%s | %.3f | " % (r["name"], n, b) + " | ".join("%.3f (%.2f)%s" % (r["new.x%s.%s" % (n, s)], b / r["new.x%s.%s" % (n, s)], "" if r["new.x%s.%s_same" % (n, s)] else " !") for s in S) + " |")
