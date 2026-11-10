#!/usr/bin/env python3
"""Round sp47: the Markdown tables of the round's section from the JSON files sp47_run.py wrote.
  tables.py ceiling ceiling.json
  tables.py ab ab.json"""
import json, math, sys


def amdahl(f, n):
    return 1 / (1 - f + f / n)


def ceiling(rows):
    print("| corpus | program | wall s | in launches | in launches of 8 blocks or more | of those, none of the static rules | bound at 2 / 4 / 8 threads | kept on one thread by |")
    print("|---|---|---:|---:|---:|---:|---|---|")
    tot = {}
    for r in rows:
        w = r["wall"]
        f = r["eligible_s"] / w
        note = ", ".join("%s %.2f s" % (k, v) for k, v in sorted(r["not_eligible"].items()))
        if r["timeout"]:
            note = (note + "; " if note else "") + "stopped at the time limit"
        elif r["rc"] != 0:
            note = (note + "; " if note else "") + "exit code %d" % r["rc"]
        print("| %s | %s | %.2f | %.1f %% | %.1f %% | %.1f %% | %.2f / %.2f / %.2f | %s |" % (
            r["corpus"], r["name"], w, 100 * r["launch_s"] / w, 100 * r["big_s"] / w, 100 * f, amdahl(f, 2), amdahl(f, 4), amdahl(f, 8), note))
        if not r["timeout"] and r["rc"] == 0:
            t = tot.setdefault(r["corpus"], [0, 0, 0, 0, 0])
            t[0] += w; t[1] += r["launch_s"]; t[2] += r["big_s"]; t[3] += r["eligible_s"]; t[4] += 1
    print()
    print("| corpus (programs that ended by themselves with exit code 0) | wall s | in launches | 8 blocks or more | none of the static rules | bound at 2 / 4 / 8 threads |")
    print("|---|---:|---:|---:|---:|---|")
    for c, t in tot.items():
        f = t[3] / t[0]
        print("| %s (%d) | %.1f | %.1f %% | %.1f %% | %.1f %% | %.2f / %.2f / %.2f |" % (
            c, t[4], t[0], 100 * t[1] / t[0], 100 * t[2] / t[0], 100 * f, amdahl(f, 2), amdahl(f, 4), amdahl(f, 8)))


def ab(rows):
    vs = [v for v in ("t1", "x1", "x2", "x4", "x8", "t8") if v in rows[0]]
    head = {"t1": "`VGPU_THREADS=1` s", "x2": "exact 2", "x4": "exact 4", "x8": "exact 8", "t8": "`VGPU_THREADS=8`"}
    print("| corpus | program | " + " | ".join(head[v] for v in vs) + " | trials kept / launches tried (8 threads) | bits |")
    print("|---|---|" + "---:|" * len(vs) + "---|---|")
    geo = {}
    for r in rows:
        cells = ["%.2f" % r["t1"]]
        for v in vs[1:]:
            cells.append("%.2f (%.2fx)" % (r[v], r["t1"] / r[v]))
            geo.setdefault((r["corpus"], v), []).append(r["t1"] / r[v])
            geo.setdefault(("all", v), []).append(r["t1"] / r[v])
        tr = r.get("x8_trials") or r.get("x4_trials") or r.get("x2_trials") or [0, 0]
        same = all(r[v + "_same"] for v in vs if v.startswith("x"))
        t8 = "" if "t8" not in r or r["t8_same"] else "; `VGPU_THREADS=8` differs"
        print("| %s | %s | %s | %d / %d | %s%s |" % (r["corpus"], r["name"], " | ".join(cells), tr[1], tr[0], "same" if same else "DIFFER", t8))
    print()
    print("| geometric mean of the speed-ups | " + " | ".join(head[v] for v in vs[1:]) + " |")
    print("|---|" + "---:|" * (len(vs) - 1))
    for c in sorted(set(k[0] for k in geo)):
        print("| %s (%d) | " % (c, len(geo[(c, vs[1])])) + " | ".join(
            "%.2fx" % math.exp(sum(map(math.log, geo[(c, v)])) / len(geo[(c, v)])) for v in vs[1:]) + " |")


if __name__ == "__main__":
    (ceiling if sys.argv[1] == "ceiling" else ab)(json.load(open(sys.argv[2])))
