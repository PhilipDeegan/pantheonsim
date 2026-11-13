import sys, os, json, time, subprocess
sys.path.insert(0, os.path.expanduser("~/.cache/sp46/tools/bench/real"))
import real
corp, build, out, cpu, tmo = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
res = {}
for (c, n, cwd, argv, copy) in real.manifest(corp, ["polybench","rodinia","hecbench","samples"]):
    while float(open("/proc/loadavg").read().split()[0]) > 8: time.sleep(10)
    r = subprocess.run(["flock", "/tmp/pantheonsim-builder.lock", "nice", "-n", "19", "python3", "-c",
        "import sys,json;sys.path.insert(0,sys.argv[1]);import real;w,rc,to,t,e=real.run_one(sys.argv[2],sys.argv[3],json.loads(sys.argv[4]),sys.argv[5]=='1',sys.argv[6],1,int(sys.argv[7]));print(json.dumps([w,rc,to]))",
        os.path.expanduser("~/.cache/sp46/tools/bench/real"), build, cwd, json.dumps(argv), "1" if copy else "0", cpu, str(tmo)],
        capture_output=True, text=True)
    try: w, rc, to = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception: w, rc, to = -1, -1, False
    res[c + "/" + n] = {"sim_s": w, "rc": rc, "timeout": to}
    print(c, n, round(w, 2), rc, to, flush=True)
    json.dump(res, open(out, "w"), indent=1)
