"""Round sp54: the programs (the corpora of tools/bench/real plus the binary64 candidates copied from the HeCBench and CUDA Samples builds),
their environment, and one run. Scratch root: ~/.cache/sp54 on the measuring host (S)."""
import os, subprocess, sys, tempfile, shutil, time
S = os.path.expanduser("~/.cache/sp54")
sys.path.insert(0, S + "/src/tools/bench/real")
import real

def manifest():
    """-> {id: (cwd, argv, copy_cwd)}; ids are corpus/name. progs2.txt lines: id|cwd under corp2|program and arguments."""
    m = {c + "/" + n: (cwd, argv, cp) for c, n, cwd, argv, cp in real.manifest(S + "/corp", ["polybench", "rodinia", "hecbench"])}
    for line in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "progs2.txt")):
        line = line.strip()
        if not line or line.startswith("#"): continue
        pid, cwd, cmd = line.split("|", 2)
        cwd = os.path.join(S, "corp2", cwd)
        a = cmd.split()
        m[pid] = (cwd, [os.path.join(cwd, a[0])] + a[1:], False)
    return m

def env(var, threads=1, extra=None):
    """A variant is <dir under S/so>[@<VGPU_SIMD value>]."""
    d, _, lvl = var.partition("@")
    e = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_") and k != "LD_PRELOAD"}
    e.update(VGPU_QUIET="1", VGPU_GPU="nvidia/rtx3060", LD_LIBRARY_PATH=S + "/so/" + d, VGPU_THREADS=str(threads))
    if lvl: e["VGPU_SIMD"] = lvl
    if extra: e.update(extra)
    return e

def run(M, prog, e, cpu, timeout=None, preload=None):
    """-> (wall seconds, exit status or None on timeout, stdout+stderr bytes)."""
    cwd, argv, cp = M[prog]
    scratch = None
    if cp:   # (Rodinia: nn wants a file list in its directory, others write files there)
        scratch = tempfile.mkdtemp(prefix="run-", dir=S)
        dd = os.path.join(os.path.dirname(argv[0]), "data")
        open(os.path.join(scratch, "filelist"), "w").write("".join(os.path.join(dd, "cane4_%d.db\n" % i) for i in range(4)))
        cwd = scratch
    cmd = ["nice", "-n", "19", "taskset", "-c", cpu]
    if preload: cmd += ["env", "LD_PRELOAD=" + preload]   # (after taskset: the sampler goes into the program, not into taskset)
    try:
        t0 = time.monotonic()
        try:
            p = subprocess.run(cmd + argv, cwd=cwd, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
            return time.monotonic() - t0, p.returncode, p.stdout
        except subprocess.TimeoutExpired as x:
            return time.monotonic() - t0, None, x.stdout or b""
    finally:
        if scratch: shutil.rmtree(scratch, ignore_errors=True)
