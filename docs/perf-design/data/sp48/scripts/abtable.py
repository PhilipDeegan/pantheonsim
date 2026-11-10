#!/usr/bin/env python3
"""abtable.py out.json name1,name2,... base_index new_index [control_index]: a markdown table of sp33.py's warm medians.
Sides are named in the order they were given to sp33.py. Prints, per program, the median wall seconds (spread = (max - min) / median) of every side,
the ratio time(base) / time(new) (above 1: new is faster), the same ratio for the control pair (two sides with the same settings: the noise floor),
and what the JIT's statistics say of the new side (warp instructions in regions, computed by generated code, region calls)."""
import json, math, sys
d = json.load(open(sys.argv[1])); names = sys.argv[2].split(","); bi, ni = int(sys.argv[3]), int(sys.argv[4])
ci = int(sys.argv[5]) if len(sys.argv) > 5 else None
print("| program | " + " | ".join("%s s (spread)" % n for n in names) + " | **%s / %s**" % (names[bi], names[ni]) + (" | control %s / %s" % (names[bi], names[ci]) if ci is not None else "") + " | in regions M | generated M | region calls M | outputs |")
print("|---|" + "---:|" * (len(names) + 1 + (1 if ci is not None else 0) + 3) + "---|")
rs = []; cs = []
for pid, r in d.items():
    w = r.get("warm")
    if not w: continue
    m, sp = w["med"], w["spread"]
    x = m[bi] / m[ni]; rs.append(x)
    j = w["samples"][ni][0].get("jit", {})
    row = "| %s | " % pid + " | ".join("%.3f (%d%%)" % (a, round(100 * b)) for a, b in zip(m, sp)) + " | **%.3f**" % x
    if ci is not None:
        c = m[bi] / m[ci]; cs.append(c); row += " | %.3f" % c
    same = len(set(tuple(h) for h in w["hashes"])) == 1
    row += " | %.1f | %.1f | %.2f | %s |" % (j.get("region_insts", 0) / 1e6, j.get("generated", 0) / 1e6, j.get("region_calls", 0) / 1e6, "same" if same else "DIFFER")
    print(row)
gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))
if rs: print("\ngeometric mean %s / %s: **%.4f** over %d programs (%.3f to %.3f)" % (names[bi], names[ni], gm(rs), len(rs), min(rs), max(rs)) + ("; control: %.4f (%.3f to %.3f)" % (gm(cs), min(cs), max(cs)) if cs else ""))
