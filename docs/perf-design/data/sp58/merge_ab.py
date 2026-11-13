#!/usr/bin/env python3
"""merge_ab.py ab.json part.json ... > merged: the rows of several sp58_ab.py outputs as one list in the order polybench, rodinia,
hecbench (a program measured twice keeps its first row), without the per-round order of the variants (kept small: the no-bloat rule)."""
import json, sys
seen, rows = set(), []
for f in sys.argv[1:]:
    for r in json.load(open(f)):
        if (r["corpus"], r["name"]) not in seen:
            seen.add((r["corpus"], r["name"]))
            r.pop("order", None)
            rows.append(r)
rows.sort(key=lambda r: ["polybench", "rodinia", "hecbench"].index(r["corpus"]))
json.dump(rows, sys.stdout, separators=(",", ":"))
