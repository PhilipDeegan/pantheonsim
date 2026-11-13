#!/usr/bin/env python3
"""prof5.py <build> <outdir> <prog,prog,...> [ENV=VAL ...]: SIGPROF sampler profile of real programs, 1 thread, cpu 14."""
import os, sys, subprocess, tempfile, shutil
S = "/tmp/sp50-avx512"
sys.path.insert(0, S + "/base/tools/bench/real")
import real
build, out, want = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
extra = dict(a.split("=", 1) for a in sys.argv[4:])
os.makedirs(out, exist_ok=True)
for corpus, name, cwd, argv, cp in real.manifest(S + "/corp", ["polybench", "rodinia", "hecbench"]):
    if corpus + "/" + name not in want:
        continue
    env = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_")}
    f = os.path.join(out, name + ".samples")
    env.update(VGPU_QUIET="1", VGPU_GPU="nvidia/rtx3060", VGPU_THREADS="1", LD_LIBRARY_PATH=build,
               LD_PRELOAD=S + "/libsigprof.so", SIGPROF_OUT=f, SIGPROF_HZ="1000")
    env.update(extra)
    scratch = None
    if cp:
        scratch = tempfile.mkdtemp(prefix="run-", dir=S)
        dd = os.path.join(os.path.dirname(argv[0]), "data")
        open(os.path.join(scratch, "filelist"), "w").write("".join(os.path.join(dd, "cane4_%d.db\n" % i) for i in range(4)))
        cwd = scratch
    r = subprocess.run(["taskset", "-c", "14"] + argv, cwd=cwd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if scratch:
        shutil.rmtree(scratch, ignore_errors=True)
    rep = subprocess.run(["python3", "-I", S + "/base/tools/bench-amd/profile/report.py", f, "--top", "400", "--exe", argv[0]], capture_output=True, text=True)
    open(os.path.join(out, name + ".report"), "w").write(rep.stdout + rep.stderr)
    print("==", corpus, name, "rc", r.returncode); print(rep.stdout[:6000])
