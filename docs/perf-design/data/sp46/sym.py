import sys, subprocess, bisect, collections, re
f = sys.argv[1]; top = int(sys.argv[2]) if len(sys.argv) > 2 else 25
maps = []; samples = []
for l in open(f):
    if l.startswith("M "):
        p = l[2:].split(); a, b = [int(x, 16) for x in p[0].split("-")]
        maps.append((a, b, int(p[2], 16), p[5] if len(p) > 5 else ""))
    elif l.startswith("S "): samples.append(int(l[2:], 16))
symc = {}
def syms(path):
    if path in symc: return symc[path]
    out = subprocess.run(["nm", "-C", "-n", "--defined-only", path], capture_output=True, text=True).stdout
    # also a PT_LOAD delta: assume vaddr==offset for text (checked separately)
    L = []
    for ln in out.splitlines():
        p = ln.split(" ", 2)
        if len(p) == 3 and p[1] in "tTwW": L.append((int(p[0], 16), p[2]))
    symc[path] = (L, [x[0] for x in L]); return symc[path]
cnt = collections.Counter(); tot = len(samples)
for ip in samples:
    for a, b, off, path in maps:
        if a <= ip < b:
            if path.endswith(".so") or ".so." in path:
                L, K = syms(path); v = ip - a + off
                i = bisect.bisect_right(K, v) - 1
                name = L[i][1] if i >= 0 else "?"
                name = re.sub(r"\((?!anonymous).*", "", name)[:100]
                cnt[path.split("/")[-1] + ":" + name] += 1
            else: cnt[path.split("/")[-1] or "anon"] += 1
            break
    else: cnt["unmapped"] += 1
print("samples", tot)
for k, v in cnt.most_common(top): print("%6.2f%% %s" % (100.0 * v / tot, k))
