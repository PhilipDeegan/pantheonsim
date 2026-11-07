#!/usr/bin/env python3
"""Launch-level result caching on the pantheon workloads (docs/pantheon-workloads.md): off, cold, warm, and warm again with
VGPU_MEMO_VERIFY=2. The workloads loop until a wall-clock duration ends, so two runs do not issue the same launches; what
is compared is what the counters say (launches and warp instructions simulated per second, hits, verification).

  memo_pantheon.py --build B --bin DIR --out OUT [--workloads a,b] [--duration 3]
"""
import argparse, fcntl, json, os, re, shutil, signal, subprocess, tempfile, time

ap = argparse.ArgumentParser()
ap.add_argument("--build", required=True)
ap.add_argument("--bin", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--workloads", default="")
ap.add_argument("--skip", default="pcie_bandwidth,p2p_thrasher,all_reduce")
ap.add_argument("--duration", type=int, default=3)
ap.add_argument("--cpu", default="9")
ap.add_argument("--max-load", type=float, default=8.0)
ap.add_argument("--timeout", type=float, default=120)
ap.add_argument("--lock", default="/tmp/pantheonsim-builder.lock")
ap.add_argument("--budget", default="")
a = ap.parse_args()
a.build, a.bin, a.out = os.path.abspath(a.build), os.path.abspath(a.bin), os.path.abspath(a.out)
os.makedirs(a.out, exist_ok=True)


def counters(path):
    n = instr = 0
    if os.path.exists(path):
        for line in open(path):
            m = re.search(r'"instructions":(\d+)', line)
            if m:
                n += 1
                instr += int(m.group(1))
    return n, instr


def run(w, memo, work):
    env = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_")}
    cf = os.path.join(work, "counters.jsonl")
    sf = os.path.join(work, "stats.txt")
    for f in (cf, sf):
        if os.path.exists(f):
            os.remove(f)
    env.update(VGPU_QUIET="1", VGPU_GPU="nvidia/a10", VGPU_VRAM_MB="1024", VGPU_DEVICE_COUNT="1", VGPU_THREADS="1",
               LD_LIBRARY_PATH=os.path.join(a.build, "shim"), VGPU_COUNTERS_FILE=cf)
    if memo is not None:
        env.update(VGPU_MEMO="1", VGPU_MEMO_STATS_FILE=sf)
        env.update(memo)
    cmd = [os.path.join(a.bin, w), "0", str(a.duration), "2", "--kernel_loops", "2", "--warmup_iters", "1", "--grid_size", "4"]
    cmd = ["nice", "-n", "19", "taskset", "-c", a.cpu] + cmd
    while os.getloadavg()[0] > a.max_load:
        time.sleep(10)
    lf = open(a.lock, "a+")
    fcntl.flock(lf, fcntl.LOCK_EX)   # (taken here so that the run's time is the program's, not the wait for its turn)
    try:
        t0 = time.monotonic()
        p = subprocess.Popen(cmd, cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        to = False
        try:
            out, _ = p.communicate(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            out, _ = p.communicate()
            to = True
        wall = time.monotonic() - t0
    finally:
        fcntl.flock(lf, fcntl.LOCK_UN)
        lf.close()
    n, instr = counters(cf)
    st = {}
    if os.path.exists(sf):
        for line in open(sf):
            for m in re.finditer(r"(\w+)=([0-9.]+)", line):
                st[m.group(1)] = st.get(m.group(1), 0) + float(m.group(2))
            m = re.search(r"reasons=(\S+)", line)
            if m and m.group(1) != "-":
                for kv in m.group(1).split(","):
                    k, _, v = kv.rpartition(":")
                    st.setdefault("reasons", {})[k] = st.get("reasons", {}).get(k, 0) + int(v)
    text = out.decode("utf-8", "replace")
    ver = re.search(r"verification[^\n]*", text, re.I)
    return dict(wall=wall, rc=p.returncode, timeout=to, launches=n, instr=instr, stats=st, verification=ver.group(0)[:80] if ver else "")


jp = os.path.join(a.out, "pantheon.json")
res = json.load(open(jp)) if os.path.exists(jp) else {}
names = sorted(os.listdir(a.bin))
if a.workloads:
    names = [n for n in names if n in a.workloads.split(",")]
names = [n for n in names if n not in a.skip.split(",")]
for w in names:
    if w in res:
        continue
    work = tempfile.mkdtemp(prefix="pan-", dir=a.out)
    cache = os.path.join(work, "cache")
    try:
        inf = {"VGPU_MEMO_DIR": cache}
        if a.budget:
            inf["VGPU_MEMO_BUDGET"] = a.budget
        r = {"off": run(w, None, work), "cold": run(w, inf, work), "warm": run(w, inf, work),
             "verify": run(w, dict(inf, VGPU_MEMO_VERIFY="2"), work)}
        r["cache_bytes"] = sum(os.path.getsize(os.path.join(rt, f)) for rt, _, fs in os.walk(cache) for f in fs) if os.path.exists(cache) else 0
    finally:
        shutil.rmtree(work, ignore_errors=True)
    res[w] = r
    print(w, {k: (v["launches"], v["stats"].get("hits")) for k, v in r.items() if isinstance(v, dict)}, flush=True)
    json.dump(res, open(jp, "w"), indent=1)
