#!/usr/bin/env python3
"""lines.py <samples.txt> <lib> <function-substring>...: the samples of sigprof.so inside the named functions of one library, split by the instruction they fell on:
the compare-and-swap (lock cmpxchg; the sample is taken at or just after it), and the rest. Also the samples in libc by exported function."""
import bisect, collections, re, subprocess, sys
samp, lib, funcs = sys.argv[1], sys.argv[2], sys.argv[3:]
by = collections.defaultdict(list); total = 0
for l in open(samp):
    if l.startswith("#"): continue
    path, off, n = l.rsplit(None, 2); by[path].append((int(off, 16), int(n))); total += int(n)
def syms(path, dyn):
    out = subprocess.run(["nm", "-C", "--defined-only"] + (["-D"] if dyn else []) + [path], capture_output=True, text=True).stdout
    s = []
    for l in out.splitlines():
        p = l.split(None, 2)
        if len(p) == 3 and p[1] in "tTwWiI": s.append((int(p[0], 16), p[2]))
    return sorted(s)
sy = syms(lib, False); ad = [x[0] for x in sy]
print("samples %d" % total)
for f in funcs:
    cands = [(a, n) for a, n in sy if f in n]
    for a, name in cands:
        i = bisect.bisect_right(ad, a); end = ad[i] if i < len(ad) else a + 4096
        dis = subprocess.run(["objdump", "-d", "--no-show-raw-insn", "--start-address=%d" % a, "--stop-address=%d" % end, lib], capture_output=True, text=True).stdout
        ins = []
        for l in dis.splitlines():
            m = re.match(r"\s*([0-9a-f]+):\s+(.*)", l)
            if m: ins.append((int(m.group(1), 16), m.group(2)))
        cas = [k for k, (x, t) in enumerate(ins) if "lock cmpxchg" in t]
        after = set()
        for k in cas:
            after.add(ins[k][0])
            if k + 1 < len(ins): after.add(ins[k + 1][0])
        tot = at_cas = 0
        for off, n in by.get(lib, []):
            if a <= off < end:
                tot += n
                if off in after: at_cas += n
        if tot: print("%6.2f%%  %-60s at a compare-and-swap %5.2f%%, elsewhere %5.2f%%" % (100.0 * tot / total, name[:60], 100.0 * at_cas / total, 100.0 * (tot - at_cas) / total))
for path, v in by.items():
    if "libc.so" not in path: continue
    ls = syms(path, True); la = [x[0] for x in ls]; c = collections.Counter()
    for off, n in v:
        i = bisect.bisect_right(la, off) - 1
        c[ls[i][1] if i >= 0 else "?"] += n
    for name, n in c.most_common(6): print("%6.2f%%  libc %s" % (100.0 * n / total, name))
