#!/usr/bin/env python3
"""count.py --build B --corp C --progs a,b --out exits.json [--env K=V,K=V] [--work D]
Round sp52: runs each program twice with the JIT (a first run fills a private cache, the second is warm) with VGPU_JIT_STATS=1 and keeps the
statistics of the warm run: the region calls, the warp instructions in them, and why each region call ended (the "jit: exit" lines).
The caller holds /tmp/pantheonsim-builder.lock (nothing is timed here, but the box is shared)."""
import argparse, importlib.util, json, os, re, shutil, subprocess, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("real", os.path.join(HERE, "real.py"))
real = importlib.util.module_from_spec(spec); spec.loader.exec_module(real)
ap = argparse.ArgumentParser()
ap.add_argument("--build", required=True); ap.add_argument("--corp", required=True); ap.add_argument("--progs", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--env", default=""); ap.add_argument("--work", default=os.path.expanduser("~/.cache/sp52/work"))
ap.add_argument("--cpu", default="6"); ap.add_argument("--timeout", type=float, default=600)
a = ap.parse_args()
byid = {p[0] + "/" + p[1]: p for p in real.manifest(a.corp, ["polybench", "rodinia", "hecbench"])}
res = json.load(open(a.out)) if os.path.exists(a.out) else {}
extra = dict(kv.split("=", 1) for kv in a.env.split(",") if kv)
for pid in a.progs.split(","):
    if pid in res: continue
    corpus, name, cwd, argv, cp = byid[pid]
    cache = os.path.join(a.work, "cache-count-" + pid.replace("/", "_")); shutil.rmtree(cache, ignore_errors=True); os.makedirs(cache, mode=0o700)
    r = {}
    for phase in ("fill", "warm"):
        sf = os.path.join(a.work, "stats.txt")
        if os.path.exists(sf): os.remove(sf)
        scratch = None; d = cwd
        if cp:
            scratch = tempfile.mkdtemp(prefix="cnt-", dir=a.work); d = scratch
            dd = os.path.join(os.path.dirname(argv[0]), "data")
            open(os.path.join(scratch, "filelist"), "w").write("".join(os.path.join(dd, "cane4_%d.db\n" % i) for i in range(4)))
        env = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_")}
        env.update(VGPU_QUIET="1", VGPU_GPU="nvidia/rtx3060", LD_LIBRARY_PATH=a.build + "/shim", VGPU_THREADS="1", VGPU_JIT="1", VGPU_JIT_THRESHOLD="100000",
                   VGPU_JIT_SYNC="1", VGPU_JIT_CACHE=cache, VGPU_JIT_STATS="1", VGPU_JIT_STATS_FILE=sf)
        env.update(extra)
        t0 = time.monotonic()
        try:
            p = subprocess.run(["nice", "-n", "19", "taskset", "-c", a.cpu] + argv, cwd=d, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=a.timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = -999
        wall = time.monotonic() - t0
        if scratch: shutil.rmtree(scratch, ignore_errors=True)
        st = open(sf).read() if os.path.exists(sf) else ""
        d = dict(rc=rc, wall=round(wall, 2), exits={})
        for l in st.splitlines():
            m = re.search(r"(\d+) region calls ran (\d+) warp instructions \((\d+) computed by generated code\).*?(\d+) branches and register-only ops continued", l)
            if m:
                for k, v in zip(("calls", "insts", "gen", "continued"), m.groups()): d[k] = d.get(k, 0) + int(v)
            m = re.match(r"jit: exit (\S+): (\d+) \(diverged warp (\d+), nothing run (\d+)\)", l)
            if m:
                e = d["exits"].setdefault(m.group(1), [0, 0, 0])
                for i in range(3): e[i] += int(m.group(2 + i))
            m = re.match(r"jit: link (\S+): (\d+)", l)
            if m: d.setdefault("link", {})[m.group(1)] = d.setdefault("link", {}).get(m.group(1), 0) + int(m.group(2))
        r[phase] = d
    shutil.rmtree(cache, ignore_errors=True)
    res[pid] = r["warm"]; res[pid]["fill_wall"] = r["fill"]["wall"]
    json.dump(res, open(a.out, "w"), indent=1)
    w = r["warm"]; print(pid, "rc", w["rc"], "wall", w["wall"], "calls", w.get("calls"), "insts", w.get("insts"), flush=True)
