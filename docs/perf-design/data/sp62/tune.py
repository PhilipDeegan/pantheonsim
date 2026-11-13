#!/usr/bin/env python3
"""Round sp62 tuning of the two new mechanisms (VGPU_EXACT_TUNE=<demote_at>,<pairs>, a knob of the measured build only).
  tune.py --base BASE --new NEW --corp C --real tools/bench/real --sp47 data/sp47 --only REGEX --threads 4,8 --runs 5 --json tune.json
Per program and thread count: the base, and the new build at each setting, interleaved, the order drawn anew per round, the machine
lock held for one round of all variants of one program at a time. Medians of the wall seconds; the signature of sp47_run (output,
exit code, counters, final memory hash) compared with the base at VGPU_THREADS=1 in every run."""
import argparse, fcntl, hashlib, json, os, random, re, shutil, statistics, sys, tempfile, time

SETTINGS = ["0,0", "0,1", "2,1", "4,0", "4,1", "8,1"]   # demote_at (0: never), pairs


def main():
    ap = argparse.ArgumentParser()
    for k in ("base", "new", "corp", "real", "sp47", "only", "json"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--threads", default="4,8")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--cpus", default="0,2,4,6,8,10,12,14")
    ap.add_argument("--lock", default="/tmp/pantheonsim-builder.lock")
    a = ap.parse_args()
    sys.path[:0] = [a.real, a.sp47]
    import real, sp47_run
    rng = random.Random(62)
    lock = open(a.lock, "w")
    rows = []
    for e in real.manifest(a.corp, ["polybench", "rodinia", "hecbench"]):
        if not re.search(a.only, e[0] + "/" + e[1]):
            continue
        variants = [("ref", a.base, {"VGPU_THREADS": "1"})]
        for n in a.threads.split(","):
            variants.append(("base.x" + n, a.base, {"VGPU_THREADS": "1", "VGPU_EXACT_THREADS": n}))
            for s in SETTINGS:
                variants.append(("new.x%s.%s" % (n, s), a.new, {"VGPU_THREADS": "1", "VGPU_EXACT_THREADS": n, "VGPU_EXACT_TUNE": s}))
        walls = {v[0]: [] for v in variants}
        sig = {v[0]: set() for v in variants}
        for r in range(a.runs):
            order = list(variants)
            rng.shuffle(order)
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                for name, b, env0 in order:
                    d = tempfile.mkdtemp(prefix="sp62tu-")
                    env = dict(env0, VGPU_COUNTERS_FILE=d + "/c", VGPU_MEMO_FINAL_HASH=d + "/f")
                    wall, rc, to, h = sp47_run.run(real, b, e, a.cpus, env, 300)
                    rd = lambda n: open(d + "/" + n, "rb").read() if os.path.exists(d + "/" + n) else b""
                    c = re.sub(rb'"(pid|launch|seq|time_ns|wall_ns)":\s*[0-9]+,?\s*', b"", rd("c"))
                    sig[name].add((h, rc, to, hashlib.sha256(c).hexdigest()[:12], hashlib.sha256(rd("f")).hexdigest()[:12]))
                    walls[name].append(round(wall, 4))
                    shutil.rmtree(d, ignore_errors=True)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
            time.sleep(2)
        ref = sig["ref"]
        row = dict(corpus=e[0], name=e[1], runs=a.runs)
        for v in variants:
            row[v[0]] = statistics.median(walls[v[0]])
            row[v[0] + "_same"] = sig[v[0]] == ref and len(ref) == 1
        rows.append(row)
        json.dump(rows, open(a.json, "w"), indent=0)
        print(e[1], " ".join("%s %.3f%s" % (v[0], row[v[0]], "" if row[v[0] + "_same"] else "!") for v in variants), flush=True)


if __name__ == "__main__":
    main()
