#!/usr/bin/env python3
"""exittable.py exits.json [exits-link.json]: markdown tables of count.py's output.
Table 1: per program, the region calls, the warp instructions a call, and the share of the calls that ended for each reason (percent of the region calls;
reasons under 0.5 % everywhere are folded into "other"). Table 2: the share of the calls that ended in a diverged warp, and that ran nothing.
With a second file (the linked side): the region calls and instructions a call of both sides, and the ops continued inside generated code."""
import json, sys
d = json.load(open(sys.argv[1]))
order = ["split-branch", "uniform-branch-taken", "uniform-branch-not-taken", "register-only-op", "turn-budget", "run-bound", "step-budget", "store-gate", "to-compiled-region",
         "bssy-nop", "bsync", "exit", "exit-no-lane-leaves", "bar-yield-call-ret-warpsync", "branch-outside-regions", "straight-op-outside-regions", "handler-op", "interpreter-op"]
progs = [p for p in d if d[p].get("calls")]
def pct(p, k, i=0):
    e = d[p]["exits"].get(k, [0, 0, 0]); tot = sum(v[0] for v in d[p]["exits"].values())
    return 100.0 * e[i] / tot if tot else 0.0
keep = [k for k in order if any(pct(p, k) >= 0.5 for p in progs)]
print("| program | region calls M | warp instructions a call | " + " | ".join(keep) + " | other |")
print("|---|---:|---:|" + "---:|" * (len(keep) + 1))
for p in progs:
    r = d[p]; tot = sum(v[0] for v in r["exits"].values()); ran = tot - sum(v[2] for v in r["exits"].values())
    print("| %s | %.2f | %.1f | " % (p, r["calls"] / 1e6, r["insts"] / max(1, r["calls"])) + " | ".join("%.1f" % pct(p, k) for k in keep) + " | %.1f |" % (100.0 - sum(pct(p, k) for k in keep)))
print()
print("| program | calls of generated code M (with those that ran nothing) | ran nothing % | in a diverged warp % | of the diverged ones: split / uniform branch / register-only op / other % |")
print("|---|---:|---:|---:|---|")
for p in progs:
    r = d[p]; ex = r["exits"]; tot = sum(v[0] for v in ex.values()); z = sum(v[2] for v in ex.values()); dv = sum(v[1] for v in ex.values())
    g = lambda *ks: sum(ex.get(k, [0, 0, 0])[1] for k in ks)
    sp, ub, ro = g("split-branch"), g("uniform-branch-taken", "uniform-branch-not-taken"), g("register-only-op")
    f = lambda x: "%.0f" % (100.0 * x / dv) if dv else "-"
    print("| %s | %.2f | %.1f | %.1f | %s / %s / %s / %s |" % (p, tot / 1e6, 100.0 * z / max(1, tot), 100.0 * dv / max(1, tot), f(sp), f(ub), f(ro), f(dv - sp - ub - ro)))
if len(sys.argv) > 2:
    l = json.load(open(sys.argv[2]))
    print()
    print("| program | region calls M: unlinked | linked | ratio | warp instructions a call: unlinked | linked | continued inside generated code M: unlinked | linked | of them linked in a diverged warp M |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for p in progs:
        if p not in l or not l[p].get("calls"): continue
        a, b = d[p], l[p]
        print("| %s | %.2f | %.2f | %.2f | %.1f | %.1f | %.2f | %.2f | %.2f |" % (p, a["calls"] / 1e6, b["calls"] / 1e6, a["calls"] / b["calls"], a["insts"] / a["calls"], b["insts"] / b["calls"],
              a.get("continued", 0) / 1e6, b.get("continued", 0) / 1e6, b.get("link", {}).get("in-diverged-warps", 0) / 1e6))
