#!/usr/bin/env python3
"""Round sp55: one program under one build, timed and (optionally) sampled or counted.

  sp55_prof.py --build B --corp C --prog polybench/ATAX --cpus 0,2 --env VGPU_THREADS=1,VGPU_EXACT_THREADS=2
               [--runs 3] [--sigprof /path/sigprof.so --out samples.txt] [--stats stats.txt]

Prints the median wall seconds and the median user+system CPU seconds of the program (children's rusage). --sigprof preloads the
SIGPROF sampler (after `taskset env`, so the samples are the program's) for one extra run; --stats sets VGPU_EXACT_STATS for one extra
run and prints the sums of every counter over the trials. Uses tools/bench/real/real.py for the manifest."""
import argparse, collections, os, re, resource, shutil, statistics, subprocess, sys, tempfile, time


def run(entry, build, cpus, env_extra, pre=None):
    corpus, name, cwd, argv, copy_cwd = entry
    env = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_")}
    env.update(VGPU_QUIET="1", VGPU_GPU="nvidia/rtx3060", LD_LIBRARY_PATH=os.path.join(build, "shim"))
    env.update(env_extra)
    scratch = None
    if copy_cwd:
        scratch = tempfile.mkdtemp(prefix="sp55-")
        dd = os.path.join(os.path.dirname(argv[0]), "data")
        with open(os.path.join(scratch, "filelist"), "w") as f:
            f.write("".join(os.path.join(dd, "cane4_%d.db\n" % i) for i in range(4)))
        cwd = scratch
    cmd = ["taskset", "-c", cpus] + (["env"] + pre if pre else []) + argv
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.monotonic()
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p.wait()   # (no timeout: with one, the wait polls and the wall time comes in steps of 50 ms)
    wall = time.monotonic() - t0
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    if scratch:
        shutil.rmtree(scratch, ignore_errors=True)
    return wall, (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime), p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--corp", required=True)
    ap.add_argument("--real", required=True)
    ap.add_argument("--prog", required=True)
    ap.add_argument("--cpus", default="0,2")
    ap.add_argument("--env", default="")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--sigprof")
    ap.add_argument("--out")
    ap.add_argument("--stats")
    a = ap.parse_args()
    sys.path.insert(0, a.real)
    import real
    c, n = a.prog.split("/")
    entry = [e for e in real.manifest(a.corp, [c]) if e[1] == n][0]
    env = dict(kv.split("=", 1) for kv in a.env.split(",") if kv)
    walls, cpus = [], []
    for _ in range(a.runs):
        w, c_, rc = run(entry, a.build, a.cpus, env)
        walls.append(w)
        cpus.append(c_)
    print("%-28s %-44s wall %7.3f cpu %7.3f rc %d" % (a.prog, a.env, statistics.median(walls), statistics.median(cpus), rc), flush=True)
    if a.sigprof:
        run(entry, a.build, a.cpus, env, pre=["LD_PRELOAD=" + a.sigprof, "SIGPROF_OUT=" + a.out, "SIGPROF_HZ=2000"])
    if a.stats:
        if os.path.exists(a.stats):
            os.unlink(a.stats)
        e2 = dict(env)
        e2["VGPU_EXACT_STATS"] = a.stats
        run(entry, a.build, a.cpus, e2)
        tot = collections.Counter()
        ends = collections.Counter()
        if os.path.exists(a.stats):
            for line in open(a.stats):
                for f in line.rstrip("\n").split("\t")[1:]:
                    k, v = f.split("=")
                    if k == "end":
                        ends[v] += 1
                    elif k not in ("blocks", "threads", "attempt"):
                        tot[k] += int(v)
        print("   trials %s  %s" % (dict(ends), "  ".join("%s=%d" % kv for kv in tot.items())), flush=True)


if __name__ == "__main__":
    main()
