#!/usr/bin/env python3
"""tables.py ab.json: the tables of Round sp58 from sp58_ab.py's output (markdown on stdout): round sp55's tables (its tables.py,
run on the same keys), then what this round adds: the seconds of trials a coarse stop threw away, and the order check."""
import json, math, os, runpy, sys

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sp55", "tables.py"), run_name="__main__")
rows = json.load(open(sys.argv[1]))
gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")
print()
print("Seconds in trials that a coarse stop threw away (first run, this round's build; programs with any):")
print()
print("| program | wall s at 2 / 4 / 8 exact threads | thrown away s at 2 / 4 / 8 | trials thrown away at 2 / 4 / 8 | first blocks run alone? |")
print("|---|---|---|---|---|")
for r in rows:
    th = r.get("thrown", {})
    if not any(th.get("b.x%d" % n, [0])[0] > 0 for n in (2, 4, 8)):
        continue
    print("| %s | %s | %s | %s | |" % (r["name"], " / ".join("%.2f" % r["b.x%d" % n] for n in (2, 4, 8)),
                                     " / ".join("%.4f" % th.get("b.x%d" % n, [0])[0] for n in (2, 4, 8)),
                                     " / ".join("%d" % r["trials"].get("b.x%d" % n, [0, 0, 0])[2] for n in (2, 4, 8))))
print()
long = [r for r in rows if r["a.t1"] >= 0.2]
for x in ("x2", "x4", "x8"):
    slow = ["%s %.2fx" % (r["name"], r["a.t1"] / r["b." + x]) for r in long if r["a.t1"] / r["b." + x] < 1.0]
    print("Programs of at least 0.2 s slower at %s exact threads than at one, this round: %s" % (x[1:], ", ".join(slow) if slow else "none"))
    slow = ["%s %.2fx" % (r["name"], r["a.t1"] / r["a." + x]) for r in long if r["a.t1"] / r["a." + x] < 1.0]
    print("... on the base: %s" % (", ".join(slow) if slow else "none"))
kept = [r for r in long if r["trials"].get("b.x8", [0, 1])[1] == 0 and r["trials"].get("a.x8", [0, 1])[1] == 0]
print("8 exact threads over plain `VGPU_THREADS=8`, time, geometric mean: all %d programs %.2fx (base %.2fx); the %d of at least 0.2 s %.2fx (base %.2fx); the %d of those whose trials were all kept %.2fx (base %.2fx)." % (
    len(rows), gm([r["b.x8"] / r["b.t8"] for r in rows]), gm([r["a.x8"] / r["b.t8"] for r in rows]),
    len(long), gm([r["b.x8"] / r["b.t8"] for r in long]), gm([r["a.x8"] / r["b.t8"] for r in long]),
    len(kept), gm([r["b.x8"] / r["b.t8"] for r in kept]), gm([r["a.x8"] / r["b.t8"] for r in kept])))
gains = {x: [r["name"] for r in long if r["a." + x] / r["b." + x] >= 1.05] for x in ("x2", "x4", "x8")}
print("Faster than the base by 5 % or more: " + "; ".join("%s threads: %d (%s)" % (x[1:], len(v), ", ".join(v)) for x, v in gains.items()))
