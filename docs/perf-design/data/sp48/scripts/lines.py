#!/usr/bin/env python3
"""lines.py <prefix>.ctlN.script <symbol substring> [top]: the source lines of the samples that fall in functions whose (mangled) name holds the
substring, by `addr2line -i` on the DSO offset (the innermost frame that is in tier_ops.inc / exec.cpp / exec_ops.inc is reported, so an inlined kernel's
samples go to the line of the run loop that called it). Shares are of ALL samples of the process (the most frequent command of the file)."""
import collections, re, subprocess, sys
path, sub = sys.argv[1], sys.argv[2]; top = int(sys.argv[3]) if len(sys.argv) > 3 else 25
rx = re.compile(r"^\s*(\S+)\s+([0-9a-f]+)\s+(\S+?)\+0x([0-9a-f]+)\s+\((.*?)\+0x([0-9a-f]+)\)\s*$")
rows = [m.groups() for m in map(rx.match, open(path, errors="replace")) if m]
comm = collections.Counter(r[0] for r in rows).most_common(1)[0][0]
rows = [r for r in rows if r[0] == comm]
total = len(rows)
by = collections.Counter((r[4], int(r[5], 16)) for r in rows if sub in r[2])
dsos = set(d for d, _ in by)
out = collections.Counter()
for dso in dsos:
    addrs = sorted(a for d, a in by if d == dso)
    # (addr2line -i prints one line per frame, innermost first: one address a call, so that the frames of an address are known)
    for a in addrs:
        q = subprocess.run(["addr2line", "-i", "-e", dso, "0x%x" % a], capture_output=True, text=True).stdout.strip().split("\n")
        pick = q[-1]
        for fr in q:
            if "tier_ops.inc" in fr or "exec.cpp" in fr or "exec_ops.inc" in fr: pick = fr; break
        out[re.sub(r".*/src/", "src/", pick).split(" ")[0]] += by[(dso, a)]
    break
n = sum(by.values())
print("%s: %d of %d samples (%.2f %%) in functions matching %r" % (path, n, total, 100.0 * n / total, sub))
for line, c in out.most_common(top):
    print("  %6.2f %%  %s" % (100.0 * c / total, line))
