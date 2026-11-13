#!/usr/bin/env python3
"""Round sp55 A/B: the base build (round sp47's shadow) against this round's, per program.

  sp55_ab.py --base BASE --new NEW --corp C --real tools/bench/real --sp47 docs/perf-design/data/sp47 --cpus 0,2,4,6,8,10,12,14
             --runs 5 --long-runs 3 --json ab.json [--only regex] [--lock /tmp/pantheonsim-builder.lock]

Variants, interleaved run by run (one run of each makes a round; the machine lock is held for a round and let go between rounds):
  a.t1   base, VGPU_THREADS=1                      the reference of the bits
  a.t1b  base, VGPU_THREADS=1 again                the A/A control
  a.x2 a.x4 a.x8   base, VGPU_EXACT_THREADS=2/4/8
  b.t1   new, VGPU_THREADS=1                       default mode (the knob off) against the base
  b.x2 b.x4 b.x8   new, VGPU_EXACT_THREADS=2/4/8
  b.t8   new, VGPU_THREADS=8                       the unproven mode: the ceiling
Per run the signature of round sp47 (sp47_run.py: normalised output and files written, exit code, VGPU_COUNTERS_FILE,
VGPU_MEMO_FINAL_HASH) is compared with a.t1's. Medians of the wall seconds of the whole program."""
import argparse, fcntl, hashlib, json, os, re, shutil, statistics, sys, tempfile, time

X = lambda n: {"VGPU_THREADS": "1", "VGPU_EXACT_THREADS": str(n)}
VARIANTS = [("a.t1", "a", {"VGPU_THREADS": "1"}), ("a.x2", "a", X(2)), ("a.x4", "a", X(4)), ("a.x8", "a", X(8)),
            ("b.t1", "b", {"VGPU_THREADS": "1"}), ("b.x2", "b", X(2)), ("b.x4", "b", X(4)), ("b.x8", "b", X(8)),
            ("a.t1b", "a", {"VGPU_THREADS": "1"}), ("b.t8", "b", {"VGPU_THREADS": "8"})]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--corp", required=True)
    ap.add_argument("--real", required=True)
    ap.add_argument("--sp47", required=True)
    ap.add_argument("--cpus", default="0,2,4,6,8,10,12,14")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--long-runs", type=int, default=3)
    ap.add_argument("--long", type=float, default=4.0, help="seconds of the first a.t1 run from which a program gets --long-runs")
    ap.add_argument("--timeout", type=float, default=150)
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--variants", default="")
    ap.add_argument("--extra-env", default="", help="NAME=value,... added to the b.x* variants")
    ap.add_argument("--lock", default="")
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.real)
    sys.path.insert(0, a.sp47)
    import real, sp47_run
    builds = {"a": a.base, "b": a.new}
    variants = [v for v in VARIANTS if not a.variants or v[0] in a.variants.split(",")]
    extra = dict(kv.split("=", 1) for kv in a.extra_env.split(",") if kv)
    progs = [e for e in real.manifest(a.corp, ["polybench", "rodinia", "hecbench"])
             if re.search(a.only, e[0] + "/" + e[1]) and not (a.skip and re.search(a.skip, e[0] + "/" + e[1]))]
    rows = json.load(open(a.json)) if os.path.exists(a.json) else []
    done = {(r["corpus"], r["name"]) for r in rows}
    lock = open(a.lock, "w") if a.lock else None
    for e in progs:
        if (e[0], e[1]) in done:
            continue
        walls = {v[0]: [] for v in variants}
        sig = {v[0]: set() for v in variants}
        trials = {}
        runs = a.runs
        r = 0
        while r < runs:
            if lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                for vname, b, venv in variants:
                    d = tempfile.mkdtemp(prefix="sp55ab-")
                    env = dict(venv)
                    if vname.startswith("b.x"):
                        env.update(extra)
                    env.update(VGPU_COUNTERS_FILE=os.path.join(d, "counters"), VGPU_MEMO_FINAL_HASH=os.path.join(d, "final"),
                               VGPU_EXACT_LOG=os.path.join(d, "exact"))
                    wall, rc, to, h = sp47_run.run(real, builds[b], e, a.cpus, env, a.timeout)

                    def rd(n):
                        try:
                            return open(os.path.join(d, n), "rb").read()
                        except OSError:
                            return b""
                    counters = re.sub(rb'"(pid|launch|seq|time_ns|wall_ns)":\s*[0-9]+,?\s*', b"", rd("counters"))
                    sig[vname].add((h, rc, to, hashlib.sha256(counters).hexdigest()[:12], hashlib.sha256(rd("final")).hexdigest()[:12]))
                    if ".x" in vname and r == 0:
                        lines = rd("exact").decode().splitlines()
                        trials[vname] = [sum(1 for l in lines if "\tcommitted\t" in l), sum(1 for l in lines if "\tstopped\t" in l),
                                         sum(1 for l in lines if "\tagain\t" in l), sum(1 for l in lines if "\tin order\t" in l)]
                    shutil.rmtree(d, ignore_errors=True)
                    walls[vname].append(wall)
            finally:
                if lock:
                    fcntl.flock(lock, fcntl.LOCK_UN)
            if r == 0 and walls[variants[0][0]][0] >= a.long:
                runs = min(runs, a.long_runs)
            r += 1
            time.sleep(0.2)   # (a waiting holder of the lock gets in)
        ref = sig[variants[0][0]]
        row = dict(corpus=e[0], name=e[1], runs=runs, ref_one=len(ref) == 1)
        for vname, _, _ in variants:
            row[vname] = statistics.median(walls[vname])
            row[vname + "_all"] = walls[vname]
            row[vname + "_same"] = sig[vname] == ref and len(ref) == 1
        row["trials"] = trials   # committed, stopped, again, in order (first run)
        rows.append(row)
        json.dump(rows, open(a.json, "w"), indent=1)
        print("%-10s %-22s " % (e[0], e[1]) + " ".join("%s %.2f%s" % (v[0], row[v[0]], "" if row[v[0] + "_same"] else "!") for v in variants)
              + "  " + " ".join("%s%s" % (k[2:], v) for k, v in trials.items()), flush=True)


if __name__ == "__main__":
    main()
