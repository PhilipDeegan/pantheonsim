#!/usr/bin/env python3
"""ab.py --variants base,base2,work,work@avx2 --progs polybench/LU,... --reps 5 --out f.json
Interleaved wall timing (order alternates per round), 1 thread, one cpu, the shared builder lock held for each round (timing) or program (--same), waits while load > --max-load.
A variant is <dir under /tmp/sp50-avx512/so>[@<VGPU_SIMD value>].
ab.py --same --ref base --variants ... : counters file, kernel digest file, rc and filtered stdout equal to the reference (threads 1 and 4)."""
import argparse, subprocess, time, os, sys, json, statistics, fcntl, re, hashlib, tempfile, shutil
S = "/tmp/sp50-avx512"
sys.path.insert(0, S + "/base/tools/bench/real")
import real
M = {c + "/" + n: (cwd, argv, cp) for c, n, cwd, argv, cp in real.manifest(S + "/corp", ["polybench", "rodinia", "hecbench"])}
def env(var, threads=1, extra=None):
    d, _, lvl = var.partition("@")
    e = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_")}
    e.update(VGPU_QUIET="1", VGPU_GPU="nvidia/rtx3060", LD_LIBRARY_PATH=S + "/so/" + d, VGPU_THREADS=str(threads))
    if lvl: e["VGPU_SIMD"] = lvl
    if extra: e.update(extra)
    return e
def wait_load(mx):
    n = 0
    while float(open("/proc/loadavg").read().split()[0]) > mx:
        time.sleep(15); n += 1
        if n % 40 == 0: print("  (waiting: load %s)" % open("/proc/loadavg").read().split()[0], flush=True)
def run_locked(prog, e, cpu, mx):
    cwd, argv, cp = M[prog]
    scratch = None
    if cp:
        scratch = tempfile.mkdtemp(prefix="run-", dir=S)
        dd = os.path.join(os.path.dirname(argv[0]), "data")
        open(os.path.join(scratch, "filelist"), "w").write("".join(os.path.join(dd, "cane4_%d.db\n" % i) for i in range(4)))
        cwd = scratch
    try:
        wait_load(mx)
        ld = open("/proc/loadavg").read().split()[0]
        t0 = time.time()
        p = subprocess.run(["nice", "-n", "19", "taskset", "-c", cpu] + argv, cwd=cwd, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return time.time() - t0, p.returncode, p.stdout, float(ld)
    finally:
        if scratch: shutil.rmtree(scratch, ignore_errors=True)
        time.sleep(0.2)
class Lock:
    def __enter__(self):
        if os.environ.get("SP50_LOCK_HELD"): return   # (the caller runs the whole script under flock: one acquisition for all rounds)
        self.lk = open("/tmp/pantheonsim-builder.lock", "a"); fcntl.flock(self.lk, fcntl.LOCK_EX)
    def __exit__(self, *a):
        if os.environ.get("SP50_LOCK_HELD"): return
        fcntl.flock(self.lk, fcntl.LOCK_UN); self.lk.close(); time.sleep(1.0)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True); ap.add_argument("--progs", required=True); ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=None); ap.add_argument("--same", action="store_true"); ap.add_argument("--ref", default="base")
    ap.add_argument("--threads", default="1,4"); ap.add_argument("--cpu", default="14"); ap.add_argument("--max-load", type=float, default=8.0)
    a = ap.parse_args()
    vs = a.variants.split(","); progs = a.progs.split(",")
    if a.same:
        bad = 0
        for pr in progs:
          with Lock():
            for th in a.threads.split(","):
                res = {}
                for v in [a.ref] + vs:
                    d = tempfile.mkdtemp(prefix="same-", dir=S)
                    cf, df = d + "/c", d + "/d"
                    dt, rc, out, ld = run_locked(pr, env(v, th, {"VGPU_COUNTERS_FILE": cf, "VGPU_KERNEL_DIGEST": df}), a.cpu if th == "1" else "8-15", 1e9)
                    txt = "\n".join(l for l in out.decode("utf-8", "replace").splitlines() if not re.search(r"[Tt]ime|sec|elapsed|GFlop|MB/s|GB/s|ms\b|^\s*[0-9.]+\s*$", l))
                    cnt = re.sub(r"\"pid\":\d+,", "", open(cf).read()) if os.path.exists(cf) else ""
                    dig = open(df).read() if os.path.exists(df) else ""
                    res[v] = (hashlib.md5(txt.encode()).hexdigest()[:8], hashlib.md5(cnt.encode()).hexdigest()[:8], hashlib.md5(dig.encode()).hexdigest()[:8], rc, len(cnt.splitlines()), len(dig.splitlines()))
                    shutil.rmtree(d, ignore_errors=True)
                r = res[a.ref]
                for v in vs:
                    ok = res[v] == r
                    bad += not ok
                    print("%-28s thr=%s %-10s %s  out/counters/digest/rc/counter-lines/digest-lines ref=%s var=%s" % (pr, th, v, "same" if ok else "DIFFERENT", r, res[v]), flush=True)
        print("DIFFERENCES:", bad)
        return
    data = {p: {v: [] for v in vs} for p in progs}
    loads = []
    for rep in range(a.reps):
      with Lock():
        for pr in progs:
            order = vs if rep % 2 == 0 else vs[::-1]
            for v in order:
                dt, rc, _, ld = run_locked(pr, env(v), a.cpu, a.max_load)
                data[pr][v].append(dt); loads.append(ld)
        if a.out: json.dump(dict(wall=data, loads=loads), open(a.out, "w"))
        print("round", rep + 1, "done", time.strftime("%T"), flush=True)
    print("load at run start: min %.1f median %.1f max %.1f" % (min(loads), statistics.median(loads), max(loads)))
    for pr in progs:
        med = {v: statistics.median(data[pr][v]) for v in vs}
        base = med[vs[0]]
        print("%-28s " % pr + "  ".join("%s %.3f (%.1f%%)" % (v, med[v], 100 * (max(data[pr][v]) - min(data[pr][v])) / med[v]) for v in vs) + "   speedup " + " ".join("%s=%.3f" % (v, base / med[v]) for v in vs[1:]), flush=True)
main()
