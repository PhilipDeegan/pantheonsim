#!/usr/bin/env python3
"""sigprof_sym.py out.txt [--top N] [--lines MODULE-SUBSTRING]: the samples of sigprof.so by module and by function (nm -C --defined-only on each module:
the function is the nearest symbol at or below the offset; a stripped or anonymous module is one row). --lines: for the modules whose path holds the
substring, the samples by offset as well (to set against objdump -d)."""
import bisect, collections, subprocess, sys
a = sys.argv[1:]; top = 25; lines = None
if "--top" in a: top = int(a[a.index("--top") + 1])
if "--lines" in a: lines = a[a.index("--lines") + 1]
mods = collections.defaultdict(list); total = 0
for l in open(a[0]):
    if l.startswith("#"): continue
    path, off, n = l.rsplit(None, 2); mods[path].append((int(off, 16), int(n))); total += int(n)
def syms(path):
    try: out = subprocess.run(["nm", "-C", "--defined-only", path], capture_output=True, text=True).stdout
    except OSError: return []
    s = []
    for l in out.splitlines():
        p = l.split(None, 2)
        if len(p) == 3 and p[1] in "tTwW": s.append((int(p[0], 16), p[2]))
    return sorted(s)
print("samples %d" % total)
rows = []
for path, v in mods.items():
    sy = syms(path) if path.startswith("/") else []
    ad = [x[0] for x in sy]; by = collections.Counter()
    for off, n in v:
        i = bisect.bisect_right(ad, off) - 1
        by[sy[i][1] if i >= 0 else "?"] += n
    msum = sum(n for _, n in v)
    print("%6.2f%%  %s" % (100.0 * msum / total, path))
    for f, n in by.items(): rows.append((n, path.rsplit("/", 1)[-1], f))
    if lines and lines in path:
        for off, n in sorted(v): print("    %8x %6d" % (off, n))
rows.sort(reverse=True)
for n, m, f in rows[:top]: print("%6.2f%%  %-28s %s" % (100.0 * n / total, m[:28], f[:150]))
