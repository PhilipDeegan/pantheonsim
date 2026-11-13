import sys, os, json, time, subprocess, tempfile, shutil, statistics
sys.path.insert(0, "/home/saqib/.cache/pantheonsim-sp46/tools/bench/real")
import real
corp = sys.argv[1]; out = sys.argv[2]; runs = int(sys.argv[3])
res = {}
for (c, n, cwd, argv, copy) in real.manifest(corp, ["polybench","rodinia","hecbench","samples"]):
    ts = []
    rc = None
    for i in range(runs):
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = "/usr/local/cuda-13.0/lib64"
        env["CUDA_VISIBLE_DEVICES"] = "0"
        scratch = None; d = cwd
        if copy:
            scratch = tempfile.mkdtemp(prefix="nat-")
            dd = os.path.join(os.path.dirname(argv[0]), "data")
            open(os.path.join(scratch, "filelist"), "w").write("".join(os.path.join(dd, "cane4_%d.db\n" % j) for j in range(4)))
            d = scratch
        t0 = time.monotonic()
        try:
            p = subprocess.run(["flock", os.path.expanduser("~/.cache/pantheonsim-card.lock")] + argv, cwd=d, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = 124
        ts.append(time.monotonic() - t0)
        if scratch: shutil.rmtree(scratch, ignore_errors=True)
    res[c + "/" + n] = {"native_s": min(ts), "rc": rc}
    print(c, n, round(min(ts), 3), rc, flush=True)
json.dump(res, open(out, "w"), indent=1)
