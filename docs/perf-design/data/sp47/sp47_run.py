#!/usr/bin/env python3
"""Round sp47 driver (docs/perf-design/rounds: "Round sp47").

  sp47_run.py ceiling --build B --corp C --cpus 8-15 --json out.json
      One run of every program with VGPU_THREADS=1 and VGPU_SP47_LOG (the build must carry ceiling-log.patch):
      per program the wall time, the time inside launches, and the part of it in launches of >= 8 blocks that
      no static rule keeps on one thread.
  sp47_run.py ab --build B --corp C --cpus 8-15 --runs 5 --json out.json [--only regex]
      The variants below, interleaved run by run, median wall seconds, and whether stdout, the files written,
      the counters file and the final memory hash are those of VGPU_THREADS=1.

Uses tools/bench/real/real.py for the corpus manifest and the output normalisation."""
import argparse, hashlib, json, os, re, statistics, subprocess, sys, tempfile, time, shutil, signal

VARIANTS = [("t1", {"VGPU_THREADS": "1"}),
            ("x2", {"VGPU_THREADS": "1", "VGPU_EXACT_THREADS": "2"}),
            ("x4", {"VGPU_THREADS": "1", "VGPU_EXACT_THREADS": "4"}),
            ("x8", {"VGPU_THREADS": "1", "VGPU_EXACT_THREADS": "8"}),
            ("t8", {"VGPU_THREADS": "8"})]


def load_wait(limit):
    while limit and os.getloadavg()[0] > limit:
        time.sleep(10)


def run(real, build, entry, cpus, env_extra, timeout):
    corpus, name, cwd, argv, copy_cwd = entry
    env = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_")}
    env.update(VGPU_QUIET="1", VGPU_GPU="nvidia/rtx3060", LD_LIBRARY_PATH=os.path.join(build, "shim"))
    env.update(env_extra)
    scratch = None
    if copy_cwd:
        scratch = tempfile.mkdtemp(prefix="sp47-")
        dd = os.path.join(os.path.dirname(argv[0]), "data")
        with open(os.path.join(scratch, "filelist"), "w") as f:
            f.write("".join(os.path.join(dd, "cane4_%d.db\n" % i) for i in range(4)))
        cwd = scratch
    t0 = time.monotonic()
    p = subprocess.Popen(["taskset", "-c", cpus] + argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, start_new_session=True)
    to = False
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
        to = True
    wall = time.monotonic() - t0
    extra = b""
    if scratch:
        for f in sorted(x for x in os.listdir(scratch) if x != "filelist"):
            fp = os.path.join(scratch, f)
            if os.path.isfile(fp) and f != "output.dat":
                extra += f.encode() + open(fp, "rb").read()
        shutil.rmtree(scratch, ignore_errors=True)
    h, checks = real.normalise(out.decode("utf-8", "replace"), extra)
    return wall, p.returncode, to, h


def ceiling(a, real, progs):
    rows = []
    for e in progs:
        load_wait(a.max_load)
        log = tempfile.mktemp(prefix="sp47log-")
        wall, rc, to, _ = run(real, a.build, e, a.cpus, {"VGPU_THREADS": "1", "VGPU_SP47_LOG": log}, a.timeout)
        tot = big = elig = 0
        n = nbig = nelig = 0
        why = {}
        if os.path.exists(log):
            for line in open(log):
                f = line.rstrip("\n").split("\t")
                # engine kernel blocks threads ns instr atomics coop clustered heap devlaunch poll smid textures externs
                blocks, ns, atomics = int(f[2]), int(f[4]), int(f[6])
                coop, clus, heap, dev, poll, smid, tex, ext = (int(x) for x in f[7:15])
                tot += ns
                n += 1
                if blocks < 8:
                    continue
                big += ns
                nbig += 1
                r = ("cooperative" if coop else "cluster" if clus else "heap/device launch" if heap or dev else
                     "polls (lockstep)" if poll else "externs" if ext else "textures" if tex else "atomics" if atomics else "")
                if r:
                    why[r] = why.get(r, 0) + ns
                else:
                    elig += ns
                    nelig += 1
            os.unlink(log)
        rows.append(dict(corpus=e[0], name=e[1], wall=wall, rc=rc, timeout=to, launches=n, launch_s=tot / 1e9,
                         big_launches=nbig, big_s=big / 1e9, eligible_launches=nelig, eligible_s=elig / 1e9,
                         not_eligible={k: v / 1e9 for k, v in why.items()}, load=os.getloadavg()[0]))
        r = rows[-1]
        print("%-10s %-24s wall %7.2f launch %7.2f big %7.2f elig %7.2f (%3.0f%%) %s" % (
            e[0], e[1], wall, r["launch_s"], r["big_s"], r["eligible_s"], 100 * r["eligible_s"] / wall if wall else 0,
            "TIMEOUT" if to else ""), flush=True)
    json.dump(rows, open(a.json, "w"), indent=1)


def ab(a, real, progs):
    names = [v for v in a.variants.split(",")]
    variants = [v for v in VARIANTS if v[0] in names]
    rows = []
    for e in progs:
        walls = {v[0]: [] for v in variants}
        sig = {v[0]: set() for v in variants}
        bad = False
        for r in range(a.runs):
            for vname, venv in variants:
                load_wait(a.max_load)
                d = tempfile.mkdtemp(prefix="sp47ab-")
                env = dict(venv)
                env.update(VGPU_COUNTERS_FILE=os.path.join(d, "counters"), VGPU_MEMO_FINAL_HASH=os.path.join(d, "final"),
                           VGPU_EXACT_LOG=os.path.join(d, "exact"))
                wall, rc, to, h = run(real, a.build, e, a.cpus, env, a.timeout)
                def rd(n):
                    try:
                        return open(os.path.join(d, n), "rb").read()
                    except OSError:
                        return b""
                counters = re.sub(rb'"(pid|launch|seq|time_ns|wall_ns)":\s*[0-9]+,?\s*', b"", rd("counters"))
                s = (h, rc, to, hashlib.sha256(counters).hexdigest()[:12], hashlib.sha256(rd("final")).hexdigest()[:12])
                ex = rd("exact").decode()
                shutil.rmtree(d, ignore_errors=True)
                walls[vname].append(wall)
                sig[vname].add(s)
                if vname.startswith("x") and r == 0:
                    lines = ex.splitlines()
                    com = sum(1 for l in lines if "\tcommitted\t" in l)
                    walls.setdefault(vname + "_trials", (len(lines), com))
                bad = bad or to
            if bad:
                break
        ref = sig[variants[0][0]]
        row = dict(corpus=e[0], name=e[1])
        for vname, _ in variants:
            row[vname] = statistics.median(walls[vname])
            row[vname + "_all"] = walls[vname]
            row[vname + "_same"] = (sig[vname] == ref and len(ref) == 1)
            if vname + "_trials" in walls:
                row[vname + "_trials"] = walls[vname + "_trials"]
        row["sig_t1"] = sorted(map(list, ref))
        rows.append(row)
        print("%-10s %-22s " % (e[0], e[1]) + "  ".join(
            "%s %7.2f%s%s" % (v, row[v], "" if row[v + "_same"] else "!DIFF",
                              (" [%d/%d]" % (row[v + "_trials"][1], row[v + "_trials"][0])) if v + "_trials" in row else "")
            for v, _ in variants), flush=True)
        json.dump(rows, open(a.json, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ceiling", "ab"])
    ap.add_argument("--build", required=True)
    ap.add_argument("--corp", required=True)
    ap.add_argument("--real", default=None, help="directory of tools/bench/real")
    ap.add_argument("--cpus", default="8-15")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=150)
    ap.add_argument("--max-load", type=float, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--corpora", default="polybench,rodinia,hecbench")
    ap.add_argument("--variants", default="t1,x2,x4,x8,t8")
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.real or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../tools/bench/real"))
    import real
    progs = [e for e in real.manifest(a.corp, a.corpora.split(",")) if re.search(a.only, e[0] + "/" + e[1])]
    (ceiling if a.mode == "ceiling" else ab)(a, real, progs)


if __name__ == "__main__":
    main()
