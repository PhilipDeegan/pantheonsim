#!/usr/bin/env python3
"""compact.py in.json out.json: sp33.py's result file with one line a program: the sides' medians, spreads, the walls of every run and whether the outputs agree
(what the tables are made of; the per-run records of return code, hash and load are dropped)."""
import json, sys
d = json.load(open(sys.argv[1])); out = []
for p, r in d.items():
    for key in ("warm", "cold"):
        w = r.get(key)
        if not w: continue
        if key == "warm":
            out.append(json.dumps({"program": p, "med": [round(x, 3) for x in w["med"]], "spread": [round(x, 3) for x in w["spread"]],
                                   "walls": [[round(x["wall"], 3) for x in s] for s in w["samples"]], "first": [round(x["wall"], 2) for x in w["first"]],
                                   "outputs_same": len(set(tuple(h) for h in w["hashes"])) == 1, "max_load": max(x["load"] for s in w["samples"] for x in s)}))
        else:
            out.append(json.dumps({"program": p, "cold": [dict(wall=round(x["wall"], 2), **x.get("jit", {})) for x in w["runs"]]}))
open(sys.argv[2], "w").write("[\n" + ",\n".join(out) + "\n]\n")
