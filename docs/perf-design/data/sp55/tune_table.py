#!/usr/bin/env python3
"""tune_table.py tune.log: the table of what each mechanism is worth (tune.sh's output) as markdown: wall and CPU seconds per program and variant."""
import collections, re, sys
X = "VGPU_THREADS=1,VGPU_EXACT_THREADS=2"
NAMES = [("VGPU_THREADS=1", "base, `VGPU_THREADS=1`"), ("VGPU_THREADS=2", "base, `VGPU_THREADS=2` (no proof)"),
         (X + "@base", "base, 2 exact threads"),
         (X + ",VGPU_EXACT_PAGES=0,VGPU_EXACT_TUNE=6", "this round's code, every page by word, no runs, no pairs"),
         (X + ",VGPU_EXACT_PAGES=0", "every page by word, with runs and pairs"),
         (X + ",VGPU_EXACT_TUNE=7,VGPU_EXACT_PAGES=2", "by page; 64 pages remembered by low bits, no runs, no pairs, no refinement"),
         (X + ",VGPU_EXACT_TUNE=1", "all but the 256 hashed pages (64 by low bits)"), (X + ",VGPU_EXACT_TUNE=2", "all but the runs of words"),
         (X + ",VGPU_EXACT_TUNE=4", "all but the pairs"), (X + ",VGPU_EXACT_PAGES=2", "all but the written-page refinement"),
         (X + "@new", "all")]
rows = collections.OrderedDict()
seen = collections.Counter()
for l in open(sys.argv[1]):
    m = re.match(r"(\S+)\s+(\S+)\s+wall\s+([\d.]+) cpu\s+([\d.]+)", l)
    if not m:
        continue
    prog, env, wall, cpu = m.group(1), m.group(2), float(m.group(3)), float(m.group(4))
    seen[(prog, env)] += 1
    key = env
    if env == X:
        key = env + ("@base" if seen[(prog, env)] == 1 else "@new")
    rows.setdefault(prog, {})[key] = (wall, cpu)
progs = list(rows)
print("| variant (2 exact threads unless said) | " + " | ".join(p.split("/")[1] for p in progs) + " |")
print("|---|" + "---:|" * len(progs))
for pat, name in NAMES:
    cells = []
    for p in progs:
        hit = [rows[p][pat]] if pat in rows[p] else []
        cells.append("%.2f / %.2f" % hit[0] if hit else "")
    print("| %s | %s |" % (name, " | ".join(cells)))
