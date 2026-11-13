#!/usr/bin/env python3
"""Round sp47: what the hooks cost a run that does not ask for a trial. BEFORE (origin/main) and AFTER (this branch), VGPU_THREADS=1,
no VGPU_EXACT_THREADS, one cpu, alternating: user instructions and cycles (perf stat), and the counters file must be the same.

  default_cost.py --before B --after A --corp C --cpu 14 --runs 5 --json out.json"""
import argparse, hashlib, json, os, re, statistics, subprocess, sys, tempfile

PROGS = ["polybench/ATAX", "polybench/SYRK", "polybench/3MM", "polybench/JACOBI2D", "polybench/MVT", "polybench/LU", "rodinia/hotspot", "rodinia/gaussian",
         "rodinia/lavaMD", "rodinia/needle", "hecbench/shmembench-cuda", "hecbench/s3d-cuda", "hecbench/aobench-cuda", "hecbench/atomicPerf-cuda"]


def main():
    ap = argparse.ArgumentParser()
    for k in ("before", "after", "corp", "json", "real"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--cpu", default="14")
    ap.add_argument("--runs", type=int, default=5)
    a = ap.parse_args()
    sys.path.insert(0, a.real)
    import real
    man = {e[0] + "/" + e[1]: e for e in real.manifest(a.corp, ["polybench", "rodinia", "hecbench"])}
    rows = []
    for name in PROGS:
        corpus, n, cwd, argv, copy = man[name]
        res = {"before": [], "after": []}
        sig = {"before": set(), "after": set()}
        for r in range(a.runs):
            for which in ("before", "after"):
                d = tempfile.mkdtemp(prefix="sp47dc-")
                env = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_")}
                env.update(VGPU_QUIET="1", VGPU_GPU="nvidia/rtx3060", VGPU_THREADS="1", LD_LIBRARY_PATH=os.path.join(getattr(a, which), "shim"),
                           VGPU_COUNTERS_FILE=os.path.join(d, "c"))
                if copy:
                    dd = os.path.join(os.path.dirname(argv[0]), "data")
                    open(os.path.join(d, "filelist"), "w").write("".join(os.path.join(dd, "cane4_%d.db\n" % i) for i in range(4)))
                # perf_event_paranoid is 4 on the measurement hosts: perf runs through sudo and the program as the user again (the two sudo
                # processes are in the count: a few million instructions, the same on both sides).
                inner = ["env"] + ["%s=%s" % kv for kv in env.items() if kv[0].startswith(("VGPU_", "LD_LIBRARY_PATH", "PATH", "HOME"))] + \
                        ["sh", "-c", 'cd "$0" && exec "$@" >/dev/null 2>&1', d if copy else cwd, "taskset", "-c", a.cpu] + argv
                p = subprocess.run(["sudo", "-n", "perf", "stat", "-x,", "-e", "instructions:u,cycles:u", "--", "sudo", "-n", "-u", os.environ["USER"]] + inner,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                vals = {}
                for line in p.stderr.splitlines():
                    f = line.split(",")
                    if len(f) > 2 and f[0].isdigit():
                        vals[f[2]] = int(f[0])
                res[which].append((vals.get("instructions:u", 0), vals.get("cycles:u", 0)))
                c = re.sub(rb'"(pid|launch|seq|time_ns|wall_ns)":\s*[0-9]+,?\s*', b"", open(os.path.join(d, "c"), "rb").read() if os.path.exists(os.path.join(d, "c")) else b"")
                sig[which].add((p.returncode, hashlib.sha256(c).hexdigest()[:12]))
                subprocess.run(["rm", "-rf", d])
        med = lambda which, i: statistics.median(x[i] for x in res[which])
        row = dict(name=name, instr_before=med("before", 0), instr_after=med("after", 0), cycles_before=med("before", 1), cycles_after=med("after", 1),
                   counters_same=(sig["before"] == sig["after"] and len(sig["before"]) == 1), load=os.getloadavg()[0])
        rows.append(row)
        print("%-26s instructions %.4f  cycles %.4f  counters %s" % (name, row["instr_after"] / row["instr_before"], row["cycles_after"] / row["cycles_before"],
                                                                     "same" if row["counters_same"] else "DIFFER"), flush=True)
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
