#!/usr/bin/env python3
"""table18.py ab-18.json ab-18-b.json ab-rule.json: round sp53's table of the 18 programs from sp33.py's files. Sides 0 to 3 of the first two files are
JIT, JIT again (the control), linked with VGPU_JIT_LINK_BSYNC=1, linked (BSYNC the run loop's: what VGPU_JIT_LINK=1 alone is); SYR2K and SYRK are taken
from the third file (the build with the loop rule: JIT, JIT again, linked, linked with VGPU_JIT_LINK_RULE=0). Medians of whole-process wall seconds
(spread = (max - min) / median)."""
import json, math, sys
a, b, r = (json.load(open(f)) for f in sys.argv[1:4])
order = ["hecbench/black-scholes-cuda", "polybench/LU", "rodinia/gaussian", "rodinia/hotspot", "polybench/SYRK", "polybench/SYR2K", "polybench/JACOBI1D", "polybench/JACOBI2D",
         "polybench/3DCONV", "polybench/2DCONV", "rodinia/lud", "rodinia/backprop", "rodinia/pathfinder", "hecbench/bilateral-cuda", "hecbench/bh-cuda", "hecbench/silu-cuda",
         "hecbench/aop-cuda", "hecbench/bitonic-sort-cuda"]
gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))
print("| program | JIT s (spread) | JIT again s (spread) | linked s (spread) | **JIT / linked** | control JIT / JIT again | linked + BSYNC s (spread) | linked / linked + BSYNC | JIT / linked + BSYNC | outputs |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
rs, cs, bs, js = [], [], [], []
for p in order:
    rule = p in r
    src = r if rule else a if p in a else b
    if p not in src: continue
    w = src[p]["warm"]; m, sp = w["med"], w["spread"]
    f = lambda i: "%.3f (%d%%)" % (m[i], round(100 * sp[i]))
    same = len(set(tuple(h) for h in w["hashes"])) == 1
    c = m[0] / m[1]; cs.append(c)
    if rule:
        x = m[0] / m[2]; rs.append(x)
        print("| %s | %s | %s | %s | **%.3f** | %.3f | (no BSYNC in it) | - | - | %s |" % (p, f(0), f(1), f(2), x, c, "same" if same else "DIFFER"))
    else:
        x = m[0] / m[3]; rs.append(x); y = m[3] / m[2]; bs.append(y); js.append(m[0] / m[2])
        print("| %s | %s | %s | %s | **%.3f** | %.3f | %s | %.3f | %.3f | %s |" % (p, f(0), f(1), f(3), x, c, f(2), y, m[0] / m[2], "same" if same else "DIFFER"))
print("\ngeometric mean JIT / linked: **%.4f** over %d programs (%.3f to %.3f); control: %.4f (%.3f to %.3f); linked / linked + BSYNC over %d programs: %.4f (%.3f to %.3f); JIT / linked + BSYNC: %.4f" %
      (gm(rs), len(rs), min(rs), max(rs), gm(cs), min(cs), max(cs), len(bs), gm(bs), min(bs), max(bs), gm(js)))
print("\n| program | JIT s (spread) | JIT again s (spread) | linked (left unlinked by the rule) s (spread) | linked, `VGPU_JIT_LINK_RULE=0` s (spread) | JIT / linked | JIT / linked, rule off | control |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for p in r:
    w = r[p]["warm"]; m, sp = w["med"], w["spread"]
    print("| %s | %s | %.3f | %.3f | %.3f |" % (p, " | ".join("%.3f (%d%%)" % (m[i], round(100 * sp[i])) for i in range(4)), m[0] / m[2], m[0] / m[3], m[0] / m[1]))
