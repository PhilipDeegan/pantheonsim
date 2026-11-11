#!/usr/bin/env python3
"""sigprof_diff.py a.txt b.txt [--top N]: two sample files of sigprof.so side by side, by function (samples are 1 ms of CPU at the default rate),
sorted by the difference b - a. A generated object (a path that ends in <32 hex digits>.so) is one row, "generated code"."""
import bisect, collections, re, subprocess, sys
def load(path):
    mods = collections.defaultdict(list)
    for l in open(path):
        if l.startswith("#"): continue
        p, off, n = l.rsplit(None, 2); mods[p].append((int(off, 16), int(n)))
    by = collections.Counter()
    for p, v in mods.items():
        if re.search(r"/[0-9a-f]{32}\.so$", p):
            by["generated code"] += sum(n for _, n in v); continue
        sy = []
        if p.startswith("/"):
            for l in subprocess.run(["nm", "-C", "--defined-only", p], capture_output=True, text=True).stdout.splitlines():
                q = l.split(None, 2)
                if len(q) == 3 and q[1] in "tTwW": sy.append((int(q[0], 16), q[2]))
        sy.sort(); ad = [x[0] for x in sy]
        for off, n in v:
            i = bisect.bisect_right(ad, off) - 1
            by[(sy[i][1] if i >= 0 else p.rsplit("/", 1)[-1])[:110]] += n
    return by
a, b = load(sys.argv[1]), load(sys.argv[2]); top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 25
print("total %d %d (%+d)" % (sum(a.values()), sum(b.values()), sum(b.values()) - sum(a.values())))
for f in sorted(set(a) | set(b), key=lambda f: -abs(b[f] - a[f]))[:top]: print("%6d %6d %+6d  %s" % (a[f], b[f], b[f] - a[f], f))
