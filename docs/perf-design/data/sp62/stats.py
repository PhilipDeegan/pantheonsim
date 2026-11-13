#!/usr/bin/env python3
"""stats.py <build> <corp> <regex of programs> <threads> [NAME=value ...]: one run of each program at that many exact threads with
VGPU_EXACT_STATS and VGPU_EXACT_LOG, under the machine lock; prints the wall seconds and the sums of the stats fields per kernel."""
import collections, os, re, subprocess, sys, tempfile, time
sys.path.insert(0, os.path.expanduser("~/.cache/sp62/new/src/tools/bench/real"))
sys.path.insert(0, os.path.expanduser("~/.cache/sp62/new/src/docs/perf-design/data/sp47"))
import real, sp47_run
b, corp, rx, n = sys.argv[1:5]
b, corp = os.path.abspath(b), os.path.abspath(corp)
extra = dict(kv.split("=", 1) for kv in sys.argv[5:])
for e in real.manifest(corp, ["polybench", "rodinia", "hecbench"]):
    if not re.search(rx, e[0] + "/" + e[1]): continue
    d = tempfile.mkdtemp(prefix="sp62st-")
    env = {"VGPU_THREADS": "1", "VGPU_EXACT_THREADS": n, "VGPU_EXACT_STATS": d + "/st", "VGPU_EXACT_LOG": d + "/log"}
    env.update(extra)
    wall, rc, to, h = sp47_run.run(real, b, e, "0,2,4,6,8,10,12,14", env, 300)
    per = collections.defaultdict(collections.Counter)
    for line in open(d + "/st") if os.path.exists(d + "/st") else []:
        f = line.rstrip("\n").split("\t")
        k = f[0]
        per[k]["trials"] += 1
        for kv in f[1:]:
            if "=" in kv:
                a, v = kv.split("=", 1)
                if v.isdigit() and a not in ("blocks", "threads", "attempt"): per[k][a] += int(v)
                elif a == "end": per[k]["end_" + v] += 1
    ends = collections.Counter(l.split("\t")[3] for l in open(d + "/log")) if os.path.exists(d + "/log") else {}
    print("%s/%s x%s wall %.2f rc %d log %s" % (e[0], e[1], n, wall, rc, dict(ends)))
    for k, c in sorted(per.items(), key=lambda kv: -kv[1]["hooks"])[:6]:
        print("  %-40s %s" % (k[:40], " ".join("%s=%d" % kv for kv in sorted(c.items()) if kv[1])))
    subprocess.run(["rm", "-rf", d])
