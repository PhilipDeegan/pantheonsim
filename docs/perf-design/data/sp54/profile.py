#!/usr/bin/env python3
"""profile.py <variant> <outdir> [--timeout 60] [--cpu 13] [--budget 600] prog,prog,...
SIGPROF sampler (tools/bench-amd/profile/sigprof.c, 1000 Hz of CPU time) on each program, one thread; writes <outdir>/<name>.report (report.py's
function table) and appends one line a program to <outdir>/shares.txt:
  id rc wall samples kernel% path% -- kernel: the binary64 lane kernel (dop_m_k, dop_avx2_t / dop512_t, dop_nan);
  path: the kernel plus its handler and the other binary64 forms (h_dop, dcmp, f2f, i2d / d2i, the f64 tiles).
Programs with a line in shares.txt are skipped (the script is rerun until the list is done); it stops starting programs after --budget seconds.
Run under the shared lock: flock /tmp/pantheonsim-builder.lock python3 profile.py ..."""
import argparse, os, re, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sp54lib as L
ap = argparse.ArgumentParser()
ap.add_argument("variant"); ap.add_argument("out"); ap.add_argument("progs")
ap.add_argument("--timeout", type=float, default=60); ap.add_argument("--cpu", default="13"); ap.add_argument("--budget", type=float, default=600)
a = ap.parse_args()
a.out = os.path.abspath(a.out)   # (the program runs in its own directory)
M = L.manifest(); os.makedirs(a.out, exist_ok=True)
sh = os.path.join(a.out, "shares.txt")
done = {l.split()[0] for l in open(sh)} if os.path.exists(sh) else set()
KERNEL = re.compile(r"dop_m_k|dop_avx2_t|dop512_t|dop_nan")
PATH = re.compile(r"dop_m_k|dop_avx2_t|dop512_t|dop_nan|h_dop|dcmp|f2f_|i2d|d2i|f2i64|tile_f64|f64_bin|h_dset|h_dmn")
t_start = time.time()
for pr in a.progs.split(","):
    if pr in done: continue
    if time.time() - t_start > a.budget: print("budget used; rerun for the rest"); break
    name = pr.replace("/", "_")
    f = os.path.join(a.out, name + ".samples")
    if os.path.exists(f): os.remove(f)
    dt, rc, out = L.run(M, pr, L.env(a.variant, 1, {"SIGPROF_OUT": f, "SIGPROF_HZ": "1000"}), a.cpu, a.timeout, preload=L.S + "/libsigprof.so")
    n = k = p = 0
    if os.path.exists(f):
        rep = subprocess.run(["python3", "-I", L.S + "/src/tools/bench-amd/profile/report.py", f, "--top", "400", "--exe", M[pr][1][0]], capture_output=True, text=True).stdout
        open(os.path.join(a.out, name + ".report"), "w").write("\n".join(rep.splitlines()[:41]) + "\n")
        for l in rep.splitlines():
            m = re.match(r"\s*([0-9.]+)%\s+(\d+)\s+(.*)", l)
            if l.startswith("total samples"): n = int(l.split()[2])
            elif m:
                if KERNEL.search(m.group(3)): k += int(m.group(2))
                if PATH.search(m.group(3)): p += int(m.group(2))
        os.remove(f)
    line = "%-34s rc %-5s wall %7.2f samples %6d kernel %5.1f%% path %5.1f%%" % (pr, "T/O" if rc is None else rc, dt, n, 100.0 * k / max(n, 1), 100.0 * p / max(n, 1))
    print(line, flush=True); open(sh, "a").write(line + "\n")
    if rc not in (0, None): print("   ", out[-300:].decode("utf-8", "replace").replace("\n", " | "))
