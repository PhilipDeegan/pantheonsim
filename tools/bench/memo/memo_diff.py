#!/usr/bin/env python3
"""Launch-level result caching (VGPU_MEMO): the differential harness and the measurement tables.

Every program of the corpora (PolyBench/GPU, Rodinia, HeCBench, CUDA Samples; layout of tools/bench/real/pack_corpora.sh)
is run (a) with the cache off, (b) cold (empty VGPU_MEMO_DIR), (c) warm (the same directory again) and, in `correctness`
mode, (d) warm again with VGPU_MEMO_VERIFY=2 (every hit is also run for real and compared). The runs must agree on:
stdout/stderr (timing lines dropped), exit code, the files the program wrote, the VGPU_KERNEL_DIGEST file, the
VGPU_COUNTERS_FILE records (launch number and pid dropped) and the hash of all device memory at exit.

  memo_diff.py --build /tmp/b --corp /tmp/corp --out /tmp/out --mode correctness [--programs a,b] [--skip x,y]
  memo_diff.py ... --mode overhead --runs 3     # medians: off, off again (A/A), cold at the default budget, warm

Simulator side only; no GPU card is touched. Waits while the load average is above --max-load.
"""
import argparse, fcntl, hashlib, importlib.util, json, os, re, shutil, signal, statistics, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("real", os.path.join(HERE, "..", "real", "real.py"))
real = importlib.util.module_from_spec(spec)
spec.loader.exec_module(real)


def dir_bytes(d):
    n = 0
    for root, _, files in os.walk(d):
        for f in files:
            try:
                n += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return n


def sha(b):
    return hashlib.sha256(b).hexdigest()[:12]


def run(build, cwd, argv, copy_cwd, cpu, threads, timeout, memo, work, lock, nice, instrument=True):
    env = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_")}
    env.update(VGPU_QUIET="1", VGPU_GPU="nvidia/rtx3060", LD_LIBRARY_PATH=os.path.join(build, "shim"))
    if threads:
        env["VGPU_THREADS"] = str(threads)
    digest = os.path.join(work, "digest.txt")
    counters = os.path.join(work, "counters.jsonl")
    final = os.path.join(work, "final.txt")
    stats = os.path.join(work, "stats.txt")
    for f in (digest, counters, final, stats):
        if os.path.exists(f):
            os.remove(f)
    if instrument:   # (the comparisons; the overhead tables run without them, which cost time of their own)
        env.update(VGPU_KERNEL_DIGEST=digest, VGPU_COUNTERS_FILE=counters, VGPU_MEMO_FINAL_HASH=final)
    if memo is not None:
        env.update(VGPU_MEMO="1", VGPU_MEMO_STATS_FILE=stats)
        env.update(memo)
    scratch = None
    if copy_cwd:
        scratch = tempfile.mkdtemp(prefix="memo-diff-", dir=work)
        dd = os.path.join(os.path.dirname(argv[0]), "data")
        with open(os.path.join(scratch, "filelist"), "w") as f:
            f.write("".join(os.path.join(dd, "cane4_%d.db\n" % i) for i in range(4)))
        cwd = scratch
    while os.getloadavg()[0] > ARGS.max_load:
        time.sleep(10)
    cmd = ["taskset", "-c", str(cpu)] + argv
    if nice:
        cmd = ["nice", "-n", "19"] + cmd
    # The machine-wide lock is taken here, not by flock(1) in front of the program, so the time of a run is the program's and not the
    # wait for its turn.
    lf = open(lock, "a+") if lock else None
    if lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
    try:
        c0 = os.times()
        t0 = time.monotonic()
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        to = False
        try:
            out, _ = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            out, _ = p.communicate()
            to = True
        wall = time.monotonic() - t0
        c1 = os.times()
    finally:
        if lf:
            fcntl.flock(lf, fcntl.LOCK_UN)
            lf.close()
    extra = b""
    if scratch:
        for f in sorted(x for x in os.listdir(scratch) if x != "filelist"):
            fp = os.path.join(scratch, f)
            if os.path.isfile(fp) and f != "output.dat":
                extra += f.encode() + open(fp, "rb").read()
        shutil.rmtree(scratch, ignore_errors=True)
    text = out.decode("utf-8", "replace")
    h, checks = real.normalise(text, extra)

    def rd(f):
        return open(f, "rb").read() if os.path.exists(f) else b""

    cnt = re.sub(rb'"launch":\d+,"pid":\d+,', b"", rd(counters))
    st = {}
    if os.path.exists(stats):
        for line in open(stats):
            for m in re.finditer(r"(\w+)=([0-9.]+)", line):
                st[m.group(1)] = st.get(m.group(1), 0) + float(m.group(2))
            m = re.search(r"reasons=(\S+)", line)
            if m and m.group(1) != "-":
                for kv in m.group(1).split(","):
                    k, _, v = kv.rpartition(":")
                    st.setdefault("reasons", {})[k] = st.get("reasons", {}).get(k, 0) + int(v)
    return dict(wall=wall, user=c1.children_user - c0.children_user, rc=p.returncode, timeout=to, out=h, checks=checks,
                digest=sha(rd(digest)), counters=sha(cnt), final=sha(rd(final)), stats=st)


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--corp", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["correctness", "overhead"], default="correctness")
    ap.add_argument("--programs", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--corpora", default="polybench,rodinia,hecbench,samples")
    ap.add_argument("--cpu", default="9")
    ap.add_argument("--threads", default="1")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--max-load", type=float, default=8.0)
    ap.add_argument("--exe-dir", default="", help="a directory of programs built from nvidia/tests/e2e (run with no arguments; shown as e2e/<name>)")
    ap.add_argument("--memo-budget", default="", help="VGPU_MEMO_BUDGET for the overhead mode's cold and warm runs ('inf': record every miss)")
    ap.add_argument("--lock", default="/tmp/pantheonsim-builder.lock", help="flock file wrapped around every run ('' for none)")
    ap.add_argument("--nice", action="store_true", default=True)
    ARGS = ap.parse_args()
    ARGS.build = os.path.abspath(ARGS.build)
    ARGS.out = os.path.abspath(ARGS.out)
    os.makedirs(ARGS.out, exist_ok=True)
    progs = real.manifest(os.path.abspath(ARGS.corp), [c for c in ARGS.corpora.split(",") if c != "e2e"]) if ARGS.corp else []
    if ARGS.exe_dir:
        d = os.path.abspath(ARGS.exe_dir)
        for n in sorted(os.listdir(d)):
            p = os.path.join(d, n)
            if os.path.isfile(p) and os.access(p, os.X_OK) and not n.endswith((".log", ".txt", ".sh")):
                progs.append(("e2e", n, d, [p], False))
    if ARGS.programs:
        want = set(ARGS.programs.split(","))
        progs = [p for p in progs if p[0] + "/" + p[1] in want or p[1] in want]
    if ARGS.skip:
        sk = set(ARGS.skip.split(","))
        progs = [p for p in progs if p[0] + "/" + p[1] not in sk]
    results = {}
    jpath = os.path.join(ARGS.out, "%s.json" % ARGS.mode)
    if os.path.exists(jpath):
        results = json.load(open(jpath))
    threads = None if ARGS.threads == "default" else ARGS.threads
    for corpus, name, cwd, argv, cp in progs:
        key = corpus + "/" + name
        if key in results:
            continue
        work = tempfile.mkdtemp(prefix="memo-work-", dir=ARGS.out)
        cache = os.path.join(work, "cache")
        r = {}
        def go(memo):
            return run(ARGS.build, cwd, argv, cp, ARGS.cpu, threads, ARGS.timeout, memo, work, ARGS.lock or None, ARGS.nice, ARGS.mode == "correctness")
        try:
            if ARGS.mode == "correctness":
                inf = {"VGPU_MEMO_DIR": cache, "VGPU_MEMO_BUDGET": "inf"}
                r["off"] = go(None)
                if r["off"]["timeout"]:
                    r["note"] = "off timed out"
                elif r["off"]["rc"] != 0:
                    r["note"] = "fails without the cache (rc=%s)" % r["off"]["rc"]
                else:
                    r["off2"] = go(None)   # more runs without the cache: what differs between them is the program's own nondeterminism
                    r["off3"] = go(None)
                    r["cold"] = go(inf)
                    r["warm"] = go(inf)
                    r["verify"] = go(dict(inf, VGPU_MEMO_VERIFY="2"))
                    r["cache_bytes"] = dir_bytes(cache)
            else:
                reps = {"off": [], "aa": [], "cold": [], "warm": []}
                for i in range(ARGS.runs):
                    d = os.path.join(work, "c%d" % i)
                    order = ["off", "aa", "cold", "warm"] if i % 2 == 0 else ["aa", "off", "cold", "warm"]
                    for s in order:
                        if s in ("off", "aa"):
                            x = go(None)
                        else:
                            if s == "cold":
                                shutil.rmtree(d, ignore_errors=True)
                            m = {"VGPU_MEMO_DIR": d}
                            if ARGS.memo_budget:
                                m["VGPU_MEMO_BUDGET"] = ARGS.memo_budget
                            x = go(m)
                        reps[s].append(x)
                        if x["timeout"]:
                            break
                    r["cache_bytes"] = dir_bytes(d)
                    if any(x[-1]["timeout"] for x in reps.values() if x):
                        break
                r["reps"] = reps
        finally:
            shutil.rmtree(work, ignore_errors=True)
        results[key] = r
        print(key, {k: (v["wall"] if isinstance(v, dict) and "wall" in v else None) for k, v in r.items() if k != "reps"}, flush=True)
        json.dump(results, open(jpath, "w"), indent=1)


if __name__ == "__main__":
    main()
