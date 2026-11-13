#!/usr/bin/env python3
"""ab.py --variants base,base2,work,work@avx2 --progs id,id,... --reps 5 --out f.json
Whole-process wall seconds, the variants interleaved (order reversed every other round), one thread, one cpu, the shared builder lock held for
each round, and a table of medians at the end. A variant is <dir under ~/.cache/sp54/so>[@<VGPU_SIMD value>]; the first is the reference.

ab.py --same --variants base,work,work@avx2 --progs ... [--threads 1,4]: the counters file, the kernel-digest file, the exit status and the
whole output of every variant against the first one's (the lock held per program). Where outputs differ the differing lines are printed, and
the difference is called "timing lines only" when every differing line pair is equal once its numbers are blanked and the line names a time
or a rate (--same runs the reference twice, so the lines that differ between two runs of one build are shown too)."""
import argparse, fcntl, hashlib, json, os, re, shutil, statistics, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sp54lib as L
TIMING = re.compile(r"(?i)time|sec\b|seconds|\bms\b|\(s\)|\bs\b|elapsed|rate|/s\b|per second|throughput|speed|GFlop|MBytes|runtime|took|usec|\bus\b|bandwidth|clock|FOM|lookups")
NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
class Lock:
    def __enter__(self):
        if os.environ.get("SP54_LOCK_HELD"): return   # (the caller runs the whole script under flock: one acquisition for all rounds)
        self.lk = open("/tmp/pantheonsim-builder.lock", "a"); fcntl.flock(self.lk, fcntl.LOCK_EX)
    def __exit__(self, *a):
        if os.environ.get("SP54_LOCK_HELD"): return
        fcntl.flock(self.lk, fcntl.LOCK_UN); self.lk.close(); time.sleep(2.0)
def h(s): return hashlib.md5(s if isinstance(s, bytes) else s.encode()).hexdigest()[:8]
def out_diff(a, b):
    """-> (kind, [differing line pairs]) for two outputs that differ."""
    la, lb = a.decode("utf-8", "replace").splitlines(), b.decode("utf-8", "replace").splitlines()
    if len(la) != len(lb): return "LINE COUNTS DIFFER (%d, %d)" % (len(la), len(lb)), []
    pairs = [(x, y) for x, y in zip(la, lb) if x != y]
    blank = lambda l: re.sub(r"\s+", " ", NUM.sub("#", l))   # (a number printed one digit shorter is padded with one more space)
    ok = all(blank(x) == blank(y) and TIMING.search(x) for x, y in pairs)
    return ("timing lines only (%d of %d lines)" % (len(pairs), len(la)) if ok else "VALUE LINES DIFFER (%d of %d lines)" % (len(pairs), len(la))), pairs
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True); ap.add_argument("--progs", required=True); ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=None); ap.add_argument("--same", action="store_true"); ap.add_argument("--threads", default="1,4")
    ap.add_argument("--cpu", default="13"); ap.add_argument("--cpus4", default="10,11,12,13")
    a = ap.parse_args()
    vs = a.variants.split(","); progs = a.progs.split(","); M = L.manifest()
    if a.same:
        bad = 0
        for pr in progs:
          with Lock():
            for th in a.threads.split(","):
                res = {}; outs = {}
                for v in [vs[0] + "#again"] + vs:
                    d = tempfile.mkdtemp(prefix="same-", dir=L.S)
                    cf, df = d + "/c", d + "/d"
                    dt, rc, out = L.run(M, pr, L.env(v.split("#")[0], th, {"VGPU_COUNTERS_FILE": cf, "VGPU_KERNEL_DIGEST": df}), a.cpu if th == "1" else a.cpus4)
                    cnt = re.sub(r"\"pid\":\d+,", "", open(cf).read()) if os.path.exists(cf) else ""
                    dig = open(df).read() if os.path.exists(df) else ""
                    res[v] = (h(cnt), h(dig), rc, len(cnt.splitlines()), len(dig.splitlines())); outs[v] = out
                    shutil.rmtree(d, ignore_errors=True)
                ref = vs[0]
                for v in [vs[0] + "#again"] + vs[1:]:
                    same = res[v] == res[ref]
                    o, pairs = ("same", []) if outs[v] == outs[ref] else out_diff(outs[ref], outs[v])
                    okay = same and (o == "same" or o.startswith("timing"))
                    bad += not okay
                    print("%-36s thr=%s %-11s counters/digest/rc/lines %s ref=%s var=%s | output: %s" % (pr, th, v, "same" if same else "DIFFERENT", res[ref], res[v], o), flush=True)
                    for x, y in pairs[:3]: print("        ref: %s\n        var: %s" % (x[:150], y[:150]))
        print("DIFFERENCES:", bad)
        return
    data = {p: {v: [] for v in vs} for p in progs}; rcs = {}
    if a.out and os.path.exists(a.out): data = json.load(open(a.out))["wall"]
    for rep in range(a.reps):
      if all(len(data[p][v]) > rep for p in progs for v in vs): continue
      with Lock():
        for pr in progs:
            order = vs if rep % 2 == 0 else vs[::-1]
            for v in order:
                dt, rc, _ = L.run(M, pr, L.env(v), a.cpu)
                data[pr][v].append(dt); rcs[pr] = rc
                time.sleep(0.2)
        if a.out: json.dump(dict(wall=data), open(a.out, "w"))
        print("round", rep + 1, "done", time.strftime("%T"), "load", open("/proc/loadavg").read().split()[0], flush=True)
    print("| program | " + " | ".join(v + " s" for v in vs) + " | " + " | ".join("%s/%s" % (vs[0], v) for v in vs[1:]) + " |")
    for pr in progs:
        med = {v: statistics.median(data[pr][v]) for v in vs}
        print("| %s | " % pr + " | ".join("%.3f (%.1f %%)" % (med[v], 100 * (max(data[pr][v]) - min(data[pr][v])) / med[v]) for v in vs) + " | "
              + " | ".join("%.3f" % (med[vs[0]] / med[v]) for v in vs[1:]) + " |", flush=True)
main()
