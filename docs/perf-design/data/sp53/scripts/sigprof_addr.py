#!/usr/bin/env python3
"""sigprof_addr.py a.txt b.txt MODULE-SUBSTRING FUNCTION-SUBSTRING [--min N]: the samples of two runs of one binary inside one function, address by
address (offset in the module, samples in a, samples in b), for the addresses with at least N samples in either; to set against objdump -d."""
import bisect, collections, subprocess, sys
def load(path, mod):
    v = collections.Counter(); p0 = None
    for l in open(path):
        if l.startswith("#"): continue
        p, off, n = l.rsplit(None, 2)
        if mod in p: v[int(off, 16)] += int(n); p0 = p
    return v, p0
mod, fn = sys.argv[3], sys.argv[4]; mn = int(sys.argv[sys.argv.index("--min") + 1]) if "--min" in sys.argv else 3
a, pa = load(sys.argv[1], mod); b, pb = load(sys.argv[2], mod)
sy = []
for l in subprocess.run(["nm", "-C", "--defined-only", pa], capture_output=True, text=True).stdout.splitlines():
    q = l.split(None, 2)
    if len(q) == 3 and q[1] in "tTwW": sy.append((int(q[0], 16), q[2]))
sy.sort(); ad = [x[0] for x in sy]
def name(off):
    i = bisect.bisect_right(ad, off) - 1
    return sy[i][1] if i >= 0 else "?"
ta = tb = 0
for off in sorted(set(a) | set(b)):
    if fn not in name(off): continue
    ta += a[off]; tb += b[off]
    if max(a[off], b[off]) >= mn: print("%8x %5d %5d %+5d" % (off, a[off], b[off], b[off] - a[off]))
print("total %d %d" % (ta, tb))
