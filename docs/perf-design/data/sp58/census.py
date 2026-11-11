#!/usr/bin/env python3
"""census.py --build B --corp C --real tools/bench/real --sp47 docs/perf-design/data/sp47 --only REGEX [--threads 2]

Round sp58's measurement before building: one run of each program at VGPU_EXACT_THREADS=N on a build of the round's base with
census.patch, which writes a line per trial (VGPU_EXACT_CENSUS): of the 4 KiB pages noted by word, and of their 64-byte lines, how
many were touched at all, by exactly one block, written in, and written in while touched by exactly one block. Sums over the
committed trials of a program, as a markdown table."""
import argparse, os, re, sys, tempfile

ap = argparse.ArgumentParser()
ap.add_argument("--build", required=True)
ap.add_argument("--corp", required=True)
ap.add_argument("--real", required=True)
ap.add_argument("--sp47", required=True)
ap.add_argument("--only", default="")
ap.add_argument("--threads", default="2")
ap.add_argument("--cpus", default="0,2,4,6,8,10,12,14")
a = ap.parse_args()
sys.path.insert(0, a.real)
sys.path.insert(0, a.sp47)
import real, sp47_run
print("| program | trials kept | pages by word | ... one block's | pages written in | ... one block's | lines touched | ... one block's | lines written in | ... one block's |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for e in real.manifest(a.corp, ["polybench", "rodinia", "hecbench"]):
    if not re.search(a.only, e[0] + "/" + e[1]):
        continue
    f = tempfile.mktemp(prefix="sp58census-")
    sp47_run.run(real, a.build, e, a.cpus, {"VGPU_THREADS": "1", "VGPU_EXACT_THREADS": a.threads, "VGPU_EXACT_CENSUS": f}, 300)
    s, n = [0] * 8, 0
    if os.path.exists(f):
        for line in open(f):
            v = [int(x) for x in line.split()]
            if v[0]:
                n += 1
                s = [x + y for x, y in zip(s, v[1:])]
        os.remove(f)
    pc = lambda x, of: "%d (%.1f %%)" % (x, 100.0 * x / of) if of else "0"
    print("| %s | %d | %d | %s | %d | %s | %d | %s | %d | %s |" % (e[1], n, s[0], pc(s[1], s[0]), s[2], pc(s[3], s[2]), s[4], pc(s[5], s[4]), s[6], pc(s[7], s[6])), flush=True)
