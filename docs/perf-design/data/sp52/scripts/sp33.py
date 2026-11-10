#!/usr/bin/env python3
"""sp33 measurement driver (server3), as rounds sp48 and sp52 used it (sp52: the JIT statistics are off unless --stats) (NOLOCK=1: the caller holds the machine lock for the whole batch).
  sp33.py --corp C --builds base=/tmp/sp33-jit/base/build,dev=/tmp/sp33-jit/dev/build --sides A=base:tier,B=dev:jit --progs a,b --runs 5 --cpu 9 --out o.json [--cache-root D] [--cold]
A side is  <build>:<tier|jit>[:ENV=V,ENV=V].  Sides are interleaved per program (A,B then B,A ...), medians of whole-process wall.
Before the timed pairs: one counting/warm-up run of every jit side on its private cache (the cache fill), whose output is also compared.
--cold: instead of warm medians, one run per jit side on an EMPTY cache (first run), recording the compile ms."""
import fcntl, argparse, importlib.util, json, os, re, shutil, statistics, subprocess, sys, tempfile, time, signal
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("real", os.path.join(HERE, "real.py"))
real = importlib.util.module_from_spec(spec); spec.loader.exec_module(real)

def base_env(build, extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_")}
    env.update(VGPU_QUIET="1", VGPU_GPU="nvidia/rtx3060", LD_LIBRARY_PATH=build + "/shim", VGPU_THREADS="1")
    env.update(extra)
    return env

LOCKED = [bool(os.environ.get("NOLOCK"))]   # True while the caller holds the machine lock for a batch of runs (NOLOCK=1: the process that started this one holds it)

def run1(build, prog, cpu, extra, timeout, scratch_root, files=()):
    corpus, name, cwd, argv, cp = prog
    scratch = None
    if cp:
        scratch = tempfile.mkdtemp(prefix="sp33-run-", dir=scratch_root)
        dd = os.path.join(os.path.dirname(argv[0]), "data")
        with open(os.path.join(scratch, "filelist"), "w") as f:
            f.write("".join(os.path.join(dd, "cane4_%d.db\n" % i) for i in range(4)))
        cwd = scratch
    lk = None
    if not LOCKED[0]:
        lk = open("/tmp/pantheonsim-builder.lock", "a")
        fcntl.flock(lk, fcntl.LOCK_EX)   # one heavy job per server: held for this run only
    try:
        t0 = time.monotonic()
        p = subprocess.Popen(["taskset", "-c", str(cpu)] + argv, cwd=cwd, env=base_env(build, extra),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        to = False
        try:
            out, _ = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL); out, _ = p.communicate(); to = True
        wall = time.monotonic() - t0
    finally:
        if lk:
            fcntl.flock(lk, fcntl.LOCK_UN); lk.close()
            time.sleep(1.5)   # (let a waiter for the lock take it: flock is not fair to a process that asks again at once)
    extra_b = b""
    if scratch:
        for f in sorted(x for x in os.listdir(scratch) if x != "filelist"):
            fp = os.path.join(scratch, f)
            if os.path.isfile(fp) and f != "output.dat":
                extra_b += f.encode() + open(fp, "rb").read()
        shutil.rmtree(scratch, ignore_errors=True)
    text = "\n".join(l for l in out.decode("utf-8", "replace").splitlines() if not l.startswith("jit: "))
    h, ch = real.normalise(text, extra_b)
    return dict(wall=wall, rc=p.returncode, timeout=to, hash=h, load=os.getloadavg()[0]), text

class Batch:
    """Holds the machine lock for a batch of runs (one round of the interleaved sides): not idle, the runs follow each other."""
    def __enter__(self):
        self.lk = None
        if os.environ.get("NOLOCK"): return
        self.lk = open("/tmp/pantheonsim-builder.lock", "a")
        fcntl.flock(self.lk, fcntl.LOCK_EX)
        LOCKED[0] = True
    def __exit__(self, *a):
        if self.lk is None: return
        LOCKED[0] = False
        fcntl.flock(self.lk, fcntl.LOCK_UN); self.lk.close()
        time.sleep(1.5)

def wait_load(maxl):
    while os.getloadavg()[0] > maxl:
        time.sleep(15)

def parse_jit(path):
    d = dict(compiles=0, compile_ms=0.0, hits=0, failures=0, region_calls=0, region_insts=0, generated=0, units=0, regions=0, declined=0, accepted=0)
    if not os.path.exists(path): return d
    for l in open(path):
        m = re.search(r"(\d+) units \((\d+) regions\), (\d+) ready, (\d+) cache hits, (\d+) compiles \(([\d.]+) ms\), (\d+) failures.*?(\d+) region calls ran (\d+) warp instructions \((\d+) computed by generated code\).*?(\d+) declined, (\d+) accepted", l)
        if m:
            u, r, rd, h, c, ms, f, rc, ri, g, de, ac = m.groups()
            d["units"] += int(u); d["regions"] += int(r); d["hits"] += int(h); d["compiles"] += int(c); d["compile_ms"] += float(ms)
            d["failures"] += int(f); d["region_calls"] += int(rc); d["region_insts"] += int(ri); d["generated"] += int(g)
            d["declined"] += int(de); d["accepted"] += int(ac)
    return d

def parse_side(s, builds):
    parts = s.split(":")
    b, kind = parts[0], parts[1]
    extra = {}
    if len(parts) > 2 and parts[2]:
        for kv in parts[2].replace(";", ",").split(","):   # (";" separates the variables of a side given on a command line whose sides are separated by ",")
            k, _, v = kv.partition("="); extra[k] = v
    return dict(build=builds[b], kind=kind, extra=extra, name=s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corp", required=True); ap.add_argument("--builds", required=True); ap.add_argument("--sides", required=True)
    ap.add_argument("--progs", required=True); ap.add_argument("--runs", type=int, default=5); ap.add_argument("--cpu", default="9")
    ap.add_argument("--out", required=True); ap.add_argument("--max-load", type=float, default=8.0); ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--work", default="/tmp/sp33-jit/work"); ap.add_argument("--cold", action="store_true")
    ap.add_argument("--threshold", default="100000")
    ap.add_argument("--stats", action="store_true", help="VGPU_JIT_STATS=1 on the jit sides (not for timings)")
    ap.add_argument("--hold", type=float, default=600, help="seconds: a program whose measurement is estimated to take less is run in one hold of the lock")
    a = ap.parse_args()
    builds = dict(x.split("=") for x in a.builds.split(","))
    sides = [parse_side(s, builds) for s in a.sides.split(",")]
    os.makedirs(a.work, exist_ok=True)
    allp = real.manifest(a.corp, ["polybench", "rodinia", "hecbench"] + (["samples"] if os.path.exists(os.path.join(a.corp, "samples/sim_main.txt")) else []))
    byid = {p[0] + "/" + p[1]: p for p in allp}
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for pid in a.progs.split(","):
        prog = byid[pid]; safe = pid.replace("/", "_")
        r = res.get(pid, {})
        key = "cold" if a.cold else "warm"
        if key in r: continue
        envs = []
        for i, sd in enumerate(sides):
            e = dict(sd["extra"])
            if sd["kind"] == "jit":
                cache = os.path.join(a.work, "cache-%s-%d" % (safe, i))
                if a.cold or not os.path.isdir(cache):
                    shutil.rmtree(cache, ignore_errors=True); os.makedirs(cache, mode=0o700); os.chmod(cache, 0o700)
                e.update(VGPU_JIT="1", VGPU_JIT_THRESHOLD=a.threshold, VGPU_JIT_SYNC="1", VGPU_JIT_CACHE=cache)
                if a.stats: e.update(VGPU_JIT_STATS="1")   # (round sp52: off for timed runs; the statistics cost atomic adds and the exit classification per region call)
            envs.append(e)
        out = {}
        def one(i, tag, nowait=False):
            sd = sides[i]; e = dict(envs[i])
            sf = None
            if sd["kind"] == "jit" and a.stats:
                sf = os.path.join(a.work, "%s.%d.jc" % (safe, i))
                if os.path.exists(sf): os.unlink(sf)
                e["VGPU_JIT_STATS_FILE"] = sf
            if not nowait and not LOCKED[0]: wait_load(a.max_load)
            x, _ = run1(sd["build"], prog, a.cpu, e, a.timeout, a.work)
            if sf: x["jit"] = parse_jit(sf)
            return x
        if a.cold:
            runs = []
            wait_load(a.max_load)
            with Batch():
                for i, sd in enumerate(sides):
                    runs.append(one(i, "cold", nowait=True))
            out["runs"] = runs
            res.setdefault(pid, {})[key] = out
            json.dump(res, open(a.out, "w"), indent=1)
            print(pid, "cold", " ".join("%s %.2f(c%d,%.0fms)" % (sd["name"], x["wall"], x.get("jit", {}).get("compiles", 0), x.get("jit", {}).get("compile_ms", 0)) for sd, x in zip(sides, runs)), flush=True)
            continue
        # warm-up (fills each jit side's private cache; also output check)
        wait_load(a.max_load)
        samples = [[] for _ in sides]
        def rounds(ks):
            for k in ks:
                order = list(range(len(sides)))
                if k % 2: order.reverse()
                for i in order:
                    samples[i].append(one(i, "t", nowait=True))
        with Batch():
            first = [one(i, "warm-up", nowait=True) for i in range(len(sides))]
            # a program whose whole measurement is short is run in this one hold of the lock, a long one a round at a time (not idle either way: the runs follow each other)
            est = sum(x["wall"] for x in first) * a.runs
            whole = est <= a.hold
            if whole:
                rounds(range(a.runs))
        if not whole:
            for k in range(a.runs):
                wait_load(a.max_load)
                with Batch():
                    rounds([k])
        out["first"] = first; out["samples"] = samples
        out["med"] = [statistics.median(x["wall"] for x in sm) for sm in samples]
        out["spread"] = [(max(x["wall"] for x in sm) - min(x["wall"] for x in sm)) / out["med"][i] for i, sm in enumerate(samples)]
        out["hashes"] = [sorted(set(x["hash"] for x in sm + [first[i]])) for i, sm in enumerate(samples)]
        res.setdefault(pid, {})[key] = out
        json.dump(res, open(a.out, "w"), indent=1)
        print(pid, " ".join("%s %.2f(%.0f%%)" % (sd["name"], m, 100 * sp) for sd, m, sp in zip(sides, out["med"], out["spread"])),
              "hashes", [len(h) for h in out["hashes"]], "same" if len(set(tuple(h) for h in out["hashes"])) == 1 else "DIFF", flush=True)

if __name__ == "__main__": main()
