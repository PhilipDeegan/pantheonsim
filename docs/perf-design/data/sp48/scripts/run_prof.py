#!/usr/bin/env python3
"""run_prof.py --corp C --build B --out DIR --progs polybench/LU,... [--modes tier,jit]
Profiles each program with prof.sh (IBS op sampling, by micro-ops and by cycles) and prints classify.py's split."""
import argparse, importlib.util, os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("real", os.path.join(HERE, "real.py"))
real = importlib.util.module_from_spec(spec); spec.loader.exec_module(real)
ap = argparse.ArgumentParser()
ap.add_argument("--corp", required=True); ap.add_argument("--build", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--progs", required=True); ap.add_argument("--modes", default="tier,jit")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
byid = {p[0] + "/" + p[1]: p for p in real.manifest(a.corp, ["polybench", "rodinia", "hecbench"])}
for pid in a.progs.split(","):
    corpus, name, cwd, argv, cp = byid[pid]
    if cp:
        cwd = tempfile.mkdtemp(prefix="sp48-run-", dir=a.out)
    for mode in a.modes.split(","):
        pre = os.path.join(a.out, pid.replace("/", "_") + "." + mode)
        subprocess.run([os.path.join(HERE, "prof.sh"), pre, a.build, mode, cwd] + argv, check=False)
        for ctl in ("1", "0"):
            print("==", pid, mode, "ibs_op cnt_ctl=" + ctl, "(1: micro-ops, 0: cycles)", flush=True)
            subprocess.run([sys.executable, os.path.join(HERE, "classify.py"), pre + ".ctl" + ctl + ".script", "--comm", os.path.basename(argv[0]), "--insn", "--top", "14"], check=False)
        if os.path.exists(pre + ".stat"): print(open(pre + ".stat").read(), flush=True)
