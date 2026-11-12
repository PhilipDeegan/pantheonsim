#!/usr/bin/env python3
"""table.py <sp57.py result>.json [--compact out.json]: the tables of round sp57 from the driver's file.
Speed-up = median wall of side a (no VGPU_JIT) / median wall of the side; above 1 the side is faster.
--compact writes the file kept in the repository: per program the same-bits verdict, every run's wall by side (seconds), and of the fill run of
each variant [objects, bytes in the cache, compiles, compile ms, sections the share rule kept, declined]."""
import json, math, statistics, sys
NAME = {"bc": "JIT cold", "bw": "JIT warm", "cc": "+LINK cold", "cw": "+LINK warm", "dc": "+LINK+FUSE cold", "dw": "+LINK+FUSE warm"}
def gm(xs): return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")
def key(x): return (x["out"], x["cnt"], x["fin"], x["rc"])

def main():
    d = json.load(open(sys.argv[1]))
    skipped = [k for k, v in d.items() if "sides" not in v]
    d = {k: v for k, v in d.items() if "sides" in v}
    sides = [s for s in NAME if any(s in v["sides"] for v in d.values())]
    rows, comp = [], {}
    for pid, v in d.items():
        S = v["sides"]; med = {s: statistics.median(x["wall"] for x in r) for s, r in S.items()}
        akeys = set(key(x) for x in S["a"]); det = len(akeys) == 1
        diff = [s for s in S if s != "a" and any(key(x) not in akeys for x in S[s])]
        fill = {k: f for k, f in v["fill"].items()}
        if any(key(f) not in akeys for f in fill.values()): diff.append("fill")
        fb = fill.get("b", {}); st = fb.get("stats", {})
        rows.append(dict(pid=pid, med=med, det=det, diff=diff, n=v["runs"], objs=fb.get("objs", 0), compiles=st.get("compiles", 0), cms=st.get("compile_ms", 0.0),
                         cold_objs=v.get("cold_objs", {}), fill=fill, rc=S["a"][0]["rc"], to=any(x["to"] for r in S.values() for x in r)))
        comp[pid] = dict(same="default differs from itself" if not det else ("DIFFERENT " + ",".join(diff) if diff else "same"),
                         walls={s: [round(x["wall"], 3) for x in r] for s, r in S.items()},
                         fill={k: [f["objs"], f["bytes"], f["stats"].get("compiles"), f["stats"].get("compile_ms"), f["stats"].get("accepted"), f["stats"].get("declined")] for k, f in fill.items()})
        co = {k: x for k, x in v.get("cold_objs", {}).items() if any(x)}
        if co: comp[pid]["objects_after_cold_runs"] = co
    if "--compact" in sys.argv:
        json.dump(comp, open(sys.argv[sys.argv.index("--compact") + 1], "w"), separators=(",", ":"))
    print("| program | runs | default s | " + " | ".join(NAME[s] for s in sides) + " | objects (b) | compile s (b) | same bits |")
    print("|---|---:|---:|" + "---:|" * len(sides) + "---:|---:|---|")
    for r in rows:
        cells = ["%.3f" % (r["med"]["a"] / r["med"][s]) if s in r["med"] else "=JIT" for s in sides]
        sb = ("same" if not r["diff"] else "DIFFERENT: " + ",".join(r["diff"])) if r["det"] else "default differs from itself"
        if r["to"]: sb += " (TIMEOUT)"
        print("| %s | %d | %.3f | %s | %d | %.1f | %s |" % (r["pid"], r["n"], r["med"]["a"], " | ".join(cells), r["objs"], r["cms"] / 1000, sb))
    print()
    print("| suite | programs | " + " | ".join(NAME[s] for s in sides) + " |")
    print("|---|---:|" + "---:|" * len(sides))
    def ratio(r, s):   # (a program whose c and d sides were not run compiled nothing with b: its c and d are b's)
        t = s if s in r["med"] else "b" + s[1]
        return r["med"]["a"] / r["med"][t]
    groups = [(c, [r for r in rows if r["pid"].startswith(c + "/")]) for c in ("polybench", "rodinia", "hecbench")] + [("all", rows), ("programs with objects", [r for r in rows if r["objs"]]), ("programs without", [r for r in rows if not r["objs"]])]
    for c, rs in groups:
        rs = [r for r in rs if not r["to"]]
        print("| %s | %d | %s |" % (c, len(rs), " | ".join("%.3f" % gm([ratio(r, s) for r in rs]) for s in sides)))
    print()
    for s in sides:
        slow = sorted(((ratio(r, s), r["pid"]) for r in rows if not r["to"] and ratio(r, s) < 1 / 1.02))
        print("%s: %d of %d programs more than 2 %% slower than the default%s" % (NAME[s], len(slow), len(rows), ": " + ", ".join("%s %.3f" % (p, x) for x, p in slow) if slow else ""))
    print()
    nd = [r["pid"] for r in rows if not r["det"]]; df = [(r["pid"], r["diff"]) for r in rows if r["det"] and r["diff"]]
    print("same bits: %d programs, %d equal in every run of every side, %d whose default runs differ among themselves (%s), %d DIFFERENT (%s)" % (
        len(rows), sum(1 for r in rows if r["det"] and not r["diff"]), len(nd), ", ".join(nd), len(df), df))
    print("not measured (over the time limit on the default):", skipped)
    print("exit status not 0 on the default:", [(r["pid"], r["rc"]) for r in rows if r["rc"] != 0])
    print("cold runs that left objects in the cache (side: objects after each run):", {r["pid"]: r["cold_objs"] for r in rows if any(any(x) for x in r["cold_objs"].values())})
if __name__ == "__main__": main()
