#!/usr/bin/env python3
"""Markdown tables from the outputs of memo_diff.py and memo_pantheon.py.

  tables.py correctness  OUTDIR [OUTDIR...]       # correctness.json files
  tables.py overhead     OUTDIR [OUTDIR...]       # overhead.json files (any budget)
  tables.py pantheon     OUTDIR
"""
import json, os, statistics, sys


def load(kind, dirs):
    res = {}
    for d in dirs:
        p = os.path.join(d, kind + ".json")
        if os.path.exists(p):
            res.update(json.load(open(p)))
    return res


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def correctness(dirs):
    res = load("correctness", dirs)
    fields = ("rc", "out", "digest", "counters", "final")
    rows, bad, ncmp, tot_launch, tot_hit, vchk, vmis, skipped = [], 0, 0, 0, 0, 0, 0, []
    for k in sorted(res):
        v = res[k]
        if "cold" not in v:
            skipped.append((k, v.get("note", "")))
            continue
        diffs = []
        for m in ("cold", "warm", "verify"):
            for f in fields:
                if v[m][f] != v["off"][f]:
                    diffs.append("%s.%s" % (m, f))
        if v["off"]["rc"] != 0:
            diffs.append("off rc=%s" % v["off"]["rc"])
        ncmp += 1
        bad += bool(diffs)
        c, w, ve = v["cold"]["stats"], v["warm"]["stats"], v["verify"]["stats"]
        tot_launch += w.get("launches", 0)
        tot_hit += w.get("hits", 0)
        vchk += ve.get("verify_checked", 0)
        vmis += ve.get("verify_mismatch", 0)
        rows.append("| %s | %d | %d | %d | %d | %d | %s |" % (
            k, w.get("launches", 0), c.get("hits", 0), w.get("hits", 0), ve.get("verify_checked", 0),
            ve.get("verify_mismatch", 0), "identical" if not diffs else "**DIFFERS: " + ", ".join(diffs) + "**"))
    print("| program | launches | hits cold | hits warm | verified | verify mismatches | off = cold = warm = verify (stdout, files, exit code, digests, counters, final memory) |")
    print("|---|---:|---:|---:|---:|---:|---|")
    print("\n".join(rows))
    print("\n%d programs compared, %d differ; warm hits %d of %d launches (%.1f %%); %d hits verified by a real run, %d mismatches." % (
        ncmp, bad, tot_hit, tot_launch, 100.0 * tot_hit / max(1, tot_launch), vchk, vmis))
    if skipped:
        print("Not compared: " + ", ".join("%s (%s)" % s for s in skipped))


def overhead(dirs):
    res = load("overhead", dirs)
    rows = []
    tot = {"off": 0.0, "aa": 0.0, "cold": 0.0, "warm": 0.0}
    over5, over10, ncount = [], [], 0
    worst = []
    for k in sorted(res):
        v = res[k]
        reps = v.get("reps")
        if not reps or not reps.get("cold"):
            continue
        if any(x["timeout"] for s in reps.values() for x in s):
            continue
        m = {s: med([x["wall"] for x in reps[s]]) for s in reps}
        st = reps["cold"][0]["stats"]
        sw = reps["warm"][0]["stats"]
        ncount += 1
        for s in tot:
            tot[s] += m[s]
        cold_r, aa_r, warm_r = m["cold"] / m["off"], m["aa"] / m["off"], m["warm"] / m["off"]
        if cold_r > 1.05:
            over5.append((k, cold_r, aa_r, m["off"]))
        if cold_r > 1.10:
            over10.append((k, cold_r, aa_r, m["off"]))
        launches = st.get("launches", 0)
        rows.append((k, m["off"], aa_r, cold_r, warm_r, launches, st.get("recorded", 0), st.get("unrecorded", 0),
                     sw.get("hits", 0), v.get("cache_bytes") or 0, len(reps["off"])))
    print("| program | off s | A/A | cold / off | warm / off | launches | recorded (cold) | unrecorded (cold) | hits (warm) | cache bytes | runs |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print("| %s | %.3f | %.3f | %.3f | %.3f | %d | %d | %d | %d | %d | %d |" % r)
    if ncount:
        print("\n%d programs. Totals of median wall seconds: off %.1f, A/A %.1f (%+.2f %%), cold %.1f (%+.2f %% against off), warm %.1f (%.2fx off)." % (
            ncount, tot["off"], tot["aa"], 100 * (tot["aa"] / tot["off"] - 1), tot["cold"], 100 * (tot["cold"] / tot["off"] - 1),
            tot["warm"], tot["warm"] / tot["off"]))
        print("Programs more than 5 %% over off when cold: %d; more than 10 %%: %d." % (len(over5), len(over10)))
        for k, c, a, o in sorted(over5, key=lambda x: -x[1]):
            print("  %s: cold %.3f, A/A %.3f, off %.3f s" % (k, c, a, o))
        # the A/A spread says how much of that is noise
        aas = [r[2] for r in rows if r[1] >= 0.3]
        if aas:
            print("A/A spread over programs of 0.3 s or more: min %.3f, median %.3f, max %.3f." % (min(aas), med(aas), max(aas)))


def pantheon(dirs):
    res = load("pantheon", dirs)
    print("| workload | launches off | cold | warm | hits cold | hits warm | simulated warp instructions per second, warm / off | cold / off (launches) | verified | mismatches | cache bytes |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    n = vm = vc = 0
    for k in sorted(res):
        v = res[k]
        off, cold, warm, ver = v["off"], v["cold"], v["warm"], v["verify"]
        if off["timeout"] or off["rc"] not in (0,):
            print("| %s | rc=%s | | | | | | | | | |" % (k, off["rc"]))
            continue
        ips = lambda x: x["instr"] / max(1e-9, x["wall"])
        print("| %s | %d | %d | %d | %d | %d | %.2fx | %.3f | %d | %d | %d |" % (
            k, off["launches"], cold["launches"], warm["launches"], cold["stats"].get("hits", 0), warm["stats"].get("hits", 0),
            ips(warm) / max(1e-9, ips(off)), cold["launches"] / max(1, off["launches"]), ver["stats"].get("verify_checked", 0),
            ver["stats"].get("verify_mismatch", 0), v.get("cache_bytes", 0)))
        n += 1
        vc += ver["stats"].get("verify_checked", 0)
        vm += ver["stats"].get("verify_mismatch", 0)
    print("\n%d workloads; %d hits verified by a real run, %d mismatches." % (n, vc, vm))


if __name__ == "__main__":
    kind, dirs = sys.argv[1], sys.argv[2:]
    {"correctness": correctness, "overhead": overhead, "pantheon": pantheon}[kind](dirs)
