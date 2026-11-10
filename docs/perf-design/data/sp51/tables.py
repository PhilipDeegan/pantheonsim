#!/usr/bin/env python3
"""tables.py split-sass.json split-ptx.json ab-okfile.json: the Markdown tables of round sp51 from the JSON that
tools/bench/real/short.py wrote (shares are of the perf samples of the program's process; wall is the median of five
unprofiled runs)."""
import json, math, sys

COLS = ["loader", "device-init", "module-load", "sass-decode", "tier-build", "ptx-parse", "execute", "teardown",
        "sim-other", "program-host"]
DC = ["module-load", "sass-decode", "tier-build", "ptx-parse"]


def split(path, title):
    d = json.load(open(path))
    print("### " + title + "\n")
    print("| program | wall ms | samples | " + " | ".join(COLS) + " | **load+decode+compile** |")
    print("|---|---:|---:|" + "---:|" * (len(COLS) + 1))
    rows = sorted(d.items(), key=lambda kv: kv[1]["wall_ms"])
    tot = {c: 0.0 for c in COLS}
    wt = {c: 0.0 for c in COLS}
    wall = 0.0
    for name, r in rows:
        s = r["share"]
        dc = sum(s[c] for c in DC)
        print("| %s | %.1f | %d | " % (name, r["wall_ms"], r["samples"]) +
              " | ".join("%.1f" % (100 * s[c]) for c in COLS) + " | **%.1f** |" % (100 * dc))
        for c in COLS:
            tot[c] += s[c]
            wt[c] += s[c] * r["wall_ms"]
        wall += r["wall_ms"]
    n = len(rows)
    print("| mean of shares | | | " + " | ".join("%.1f" % (100 * tot[c] / n) for c in COLS) +
          " | **%.1f** |" % (100 * sum(tot[c] for c in DC) / n))
    print("| wall-weighted | %.1f | | " % wall + " | ".join("%.1f" % (100 * wt[c] / wall) for c in COLS) +
          " | **%.1f** |" % (100 * sum(wt[c] for c in DC) / wall))
    # what a cache that removed all of load+decode+compile would be worth
    g = math.exp(sum(-math.log(1 - sum(r["share"][c] for c in DC)) for _, r in rows) / n)
    over = sum(1 for _, r in rows if sum(r["share"][c] for c in DC) >= 0.15)
    print("\nCeiling (every sample of those four columns removed, nothing added): geomean %.3fx over %d programs; "
          "%d of %d programs at or above 15 %%.\n" % (g, n, over, n))


def ab(path):
    d = json.load(open(path))
    names = list(next(iter(d.values()))["median_ms"])
    print("### The verdict file there already is (`VGPU_SASS_CACHE`): off, cold, warm\n")
    print("| program | " + " | ".join(n + " ms" for n in names) + " | off/warm | off/cold | outputs |")
    print("|---|" + "---:|" * (len(names) + 2) + "---|")
    for p, r in sorted(d.items(), key=lambda kv: kv[1]["median_ms"]["warm"]):
        m = r["median_ms"]
        print("| %s | " % p + " | ".join("%.2f" % m[n] for n in names) +
              " | %.3f | %.3f | %s |" % (m["off"] / m["warm"], m["off"] / m["cold"], "same" if r["same"] else "DIFFER"))
    for n in ("warm", "cold"):
        g = math.exp(sum(math.log(r["median_ms"]["off"] / r["median_ms"][n]) for r in d.values()) / len(d))
        print("\ngeomean off/%s = %.4f over %d programs" % (n, g, len(d)))
    print()


split(sys.argv[1], "Split, SASS engine (the default), verdict file warm: percent of samples")
split(sys.argv[2], "Split, PTX engine (`VGPU_SASS=0`): percent of samples")
ab(sys.argv[3])
