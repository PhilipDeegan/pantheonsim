#!/usr/bin/env python3
"""verify.py --corp C --builds b=..,d=.. --ref b:tier --variants d:tier,d:jit,... --progs a,b [--threads 1,4]
Same bits: normalised stdout hash, VGPU_COUNTERS_FILE contents and VGPU_KERNEL_DIGEST of every variant must equal the reference's."""
import argparse, importlib.util, json, os, re, shutil, subprocess, sys, tempfile, time, fcntl, signal
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("sp33", os.path.join(HERE, "sp33.py"))
sp = importlib.util.module_from_spec(spec); spec.loader.exec_module(sp)
real = sp.real
LOCKED = [False]

def run(build, prog, extra, threads, work, timeout, cpus):
    corpus, name, cwd, argv, cp = prog
    cf = os.path.join(work, "cnt.%d" % os.getpid()); df = os.path.join(work, "dig.%d" % os.getpid())
    for f in (cf, df):
        if os.path.exists(f): os.unlink(f)
    env = sp.base_env(build, dict(extra, VGPU_THREADS=str(threads), VGPU_COUNTERS_FILE=cf, VGPU_KERNEL_DIGEST=df))
    scratch = None
    if cp:
        scratch = tempfile.mkdtemp(prefix="sp33-v-", dir=work)
        dd = os.path.join(os.path.dirname(argv[0]), "data")
        open(os.path.join(scratch, "filelist"), "w").write("".join(os.path.join(dd, "cane4_%d.db\n" % i) for i in range(4)))
        cwd = scratch
    lk = None
    if not sp.LOCKED[0]:
        lk = open("/tmp/pantheonsim-builder.lock", "a"); fcntl.flock(lk, fcntl.LOCK_EX)
    try:
        p = subprocess.Popen(["taskset", "-c", cpus] + argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        try: out, _ = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL); out, _ = p.communicate()
    finally:
        if lk:
            fcntl.flock(lk, fcntl.LOCK_UN); lk.close()
            time.sleep(1.5)
    eb = b""
    if scratch:
        for f in sorted(x for x in os.listdir(scratch) if x != "filelist"):
            fp = os.path.join(scratch, f)
            if os.path.isfile(fp) and f != "output.dat": eb += f.encode() + open(fp, "rb").read()
        shutil.rmtree(scratch, ignore_errors=True)
    h, _ = real.normalise(out.decode("utf-8", "replace"), eb)
    cnt = re.sub(r'"pid":\d+,', "", open(cf).read()) if os.path.exists(cf) else ""
    dig = open(df).read() if os.path.exists(df) else ""
    return h, cnt, dig, p.returncode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corp", required=True); ap.add_argument("--builds", required=True); ap.add_argument("--ref", required=True)
    ap.add_argument("--variants", required=True); ap.add_argument("--progs", required=True); ap.add_argument("--threads", default="1")
    ap.add_argument("--work", default="/tmp/sp33-jit/work"); ap.add_argument("--timeout", type=float, default=1200); ap.add_argument("--cpus", default="8-11")
    ap.add_argument("--threshold", default="100000"); ap.add_argument("--max-load", type=float, default=12)
    a = ap.parse_args()
    builds = dict(x.split("=") for x in a.builds.split(","))
    os.makedirs(a.work, exist_ok=True)
    allp = real.manifest(a.corp, ["polybench", "rodinia", "hecbench", "samples"])
    byid = {p[0] + "/" + p[1]: p for p in allp}
    ref = sp.parse_side(a.ref, builds); vs = [sp.parse_side(v, builds) for v in a.variants.split(",")]
    bad = 0
    for pid in a.progs.split(","):
        prog = byid[pid]
        sp.wait_load(a.max_load)
        with sp.Batch():   # every thread count, the reference and every variant of this program: one hold of the machine lock
          for th in a.threads.split(","):
            if True:
                r = run(ref["build"], prog, ref["extra"], th, a.work, a.timeout, a.cpus)
                for i, v in enumerate(vs):
                    e = dict(v["extra"])
                    if v["kind"] == "jit":
                        cache = os.path.join(a.work, "vcache-%s-%d" % (pid.replace("/", "_"), i)); shutil.rmtree(cache, ignore_errors=True); os.makedirs(cache, mode=0o700)
                        e.update(VGPU_JIT="1", VGPU_JIT_THRESHOLD=a.threshold, VGPU_JIT_SYNC="1", VGPU_JIT_CACHE=cache)
                    for phase in (("cold", "warm") if v["kind"] == "jit" else ("tier",)):
                        x = run(v["build"], prog, e, th, a.work, a.timeout, a.cpus)
                        ok = x[0] == r[0] and x[1] == r[1] and x[2] == r[2] and x[3] == r[3]
                        if not ok: bad += 1
                        print("%-34s thr=%s %-26s %-5s %s%s" % (pid, th, v["name"], phase, "same" if ok else "DIFFERENT",
                              "" if ok else " out:%s/%s cnt:%s dig:%s rc:%s/%s" % (x[0], r[0], x[1] == r[1], x[2] == r[2], x[3], r[3])), flush=True)
    print("DIFFERENCES:", bad)
    sys.exit(1 if bad else 0)
if __name__ == "__main__": main()
