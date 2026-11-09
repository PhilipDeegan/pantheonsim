#!/usr/bin/env python3
"""Measure a PGO build against a plain one on the programs of a manifest (scripts/pgo/test-manifest.txt).

  ab.py time --manifest M --builds A=/path/a,A2=/path/a-copy,P=/path/pgo [--ref A] --runs 9 --cpus 9 --corp C --py PY
  ab.py bits --manifest M --builds A=/path/a,A2=/path/a-copy,P=/path/pgo --threads-list 1,4 --corp C --py PY
  both: [--only REGEX] [--skip REGEX] [--lock FILE] [--max-load 8] [--json FILE]

time  Every program runs --warmup + --runs times per build, interleaved (the build order rotates every round so that no
      build always goes first), pinned to --cpus (one CPU for threads=1 entries, a CPU list for --threads-cpus runs).
      Table: median CPU seconds (pantheon workloads: per 1e9 warp-instructions) (user+system; wall seconds with --metric wall) per build and the ratio REF/build, so a
      ratio above 1 means the build is faster than REF. Geometric mean per group and overall. Builds named like the REF
      with a "2" suffix (A2) are the A/A control: its ratio is the noise floor.
bits  Every program runs once per build and thread count with VGPU_COUNTERS_FILE, VGPU_KERNEL_DIGEST and
      VGPU_MEMO_FINAL_HASH; the normalised output, the files the program wrote, the counters, the kernel digest, the
      hash of device memory at exit and the exit code must be equal across all builds. A program for which REF and the
      A/A copy already differ is reported as NONDETERMINISTIC, not as a PGO difference.

The wall-clock time of a run excludes the wait for --lock. Simulator side only; no GPU card is touched.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import manifest as mf  # noqa: E402

_real = None


def real():
    global _real
    if _real is None:
        p = os.path.join(HERE, "..", "..", "tools", "bench", "real", "real.py")
        spec = importlib.util.spec_from_file_location("real", p)
        _real = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_real)
    return _real


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


class Runner:
    def __init__(self, a):
        self.a = a
        self.ctx_base = {"CORP": os.path.abspath(a.corp) if a.corp else "", "SRC": os.path.abspath(a.src), "PY": a.py}

    def run(self, entry, build, cpus, threads=None, collect=None, timeout=None):
        a = self.a
        scratch = tempfile.mkdtemp(prefix="pgo-ab-")
        ctx = dict(self.ctx_base, BUILD=os.path.abspath(build), TMP=scratch)
        e = mf.Entry(**{k: getattr(entry, k) for k in mf.Entry.__slots__})
        if threads:
            e.threads = threads
        env = mf.make_env(e, ctx)
        paths = {}
        if collect:
            for k, v in (("VGPU_KERNEL_DIGEST", "digest"), ("VGPU_COUNTERS_FILE", "counters"), ("VGPU_MEMO_FINAL_HASH", "final")):
                if collect == "counters" and v != "counters":
                    continue
                paths[v] = os.path.join(scratch, v + ".txt")
                env[k] = paths[v]
        argv = [mf.subst(x, ctx) for x in e.command]
        cwd = scratch if e.cwd == "tmp" else mf.subst(e.cwd, ctx)
        if cpus:
            argv = ["taskset", "-c", cpus] + argv
        while os.getloadavg()[0] > a.max_load:
            time.sleep(10)
        lf = None
        if a.lock:
            lf = open(a.lock, "a+")
            fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            t0 = time.monotonic()
            p = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
            to = False
            try:
                out, _ = p.communicate(timeout=timeout or e.cap)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                out, _ = p.communicate()
                to = True
            wall = time.monotonic() - t0
        finally:
            if lf:
                fcntl.flock(lf, fcntl.LOCK_UN)
                lf.close()
        res = {"wall": wall, "rc": p.returncode, "timeout": to, "text": out.decode("utf-8", "replace")}
        extra = b""
        for f in sorted(os.listdir(scratch)):
            fp = os.path.join(scratch, f)
            if os.path.isfile(fp) and f not in ("digest.txt", "counters.txt", "final.txt") and f != "output.dat" and e.cwd == "tmp":
                extra += f.encode() + open(fp, "rb").read()
        if collect == "counters":
            n = 0
            for line in rd_lines(paths["counters"]):
                m = re.search(r'"instructions":(\d+)', line)
                if m:
                    n += int(m.group(1))
            res["instr"] = n
        elif collect:
            def rd(k):
                return open(paths[k], "rb").read() if os.path.exists(paths[k]) else b""
            res["digest"] = hashlib.sha256(rd("digest")).hexdigest()[:12]
            res["counters"] = hashlib.sha256(re.sub(rb'"launch":\d+,"pid":\d+,', b"", rd("counters"))).hexdigest()[:12]
            res["final"] = hashlib.sha256(rd("final")).hexdigest()[:12]
        # host timing a program prints is not a result: sweep.py's "(0.1 s)" after each check, the CUDA samples' "Time: 12.3 ms" and "TOPS: 0.8"
        txt = re.sub(r"\((\d+(\.\d+)?) s\)", "(T s)", res["text"])
        txt = re.sub(r"(Time|TOPS|TFLOPS): *\d+(\.\d+)?( ms)?", r"\1: T", txt)
        res["out"], res["checks"] = real().normalise(txt, extra)
        shutil.rmtree(scratch, ignore_errors=True)
        return res


def rd_lines(path):
    return open(path, "r", errors="replace").read().splitlines() if os.path.exists(path) else []


def cpu_children():
    # getrusage has microsecond resolution; os.times() has 10 ms ticks, which reads a 20 ms start-up program as 0.000
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def select(a):
    es = mf.parse(a.manifest)
    if a.only:
        es = [e for e in es if re.search(a.only, e.id)]
    if a.skip:
        es = [e for e in es if not re.search(a.skip, e.id)]
    return es


def parse_builds(s):
    out = {}
    for item in s.split(","):
        k, v = item.split("=", 1)
        out[k] = os.path.abspath(v)
    return out


def cmd_time(a):
    builds = parse_builds(a.builds)
    names = list(builds)
    ref = a.ref or names[0]
    rn = Runner(a)
    rows = []
    for e in select(a):
        multi = e.threads > 1 or a.threads > 1
        thr = a.threads if a.threads > 1 else None
        cpus = a.threads_cpus if multi else a.cpus
        samples = {n: [] for n in names}
        walls = {n: [] for n in names}
        bad = []
        nrun = a.runs + a.warmup
        for r in range(nrun):
            order = names[r % len(names):] + names[:r % len(names)]
            for n in order:
                c0 = cpu_children()
                # a program that runs for a duration (the pantheon workloads) does a different amount of work per run:
                # its figure is CPU seconds per 1e9 warp-instructions executed
                res = rn.run(e, builds[n], cpus, thr, collect="counters" if e.id.startswith("pan/") else None)
                c1 = cpu_children()
                if res["rc"] != 0 or res["timeout"]:
                    bad.append("%s rc=%s%s" % (n, res["rc"], " TIMEOUT" if res["timeout"] else ""))
                    continue
                if r >= a.warmup:
                    work = res["instr"] / 1e9 if "instr" in res else 1.0
                    if work <= 0:
                        bad.append("%s no instructions counted" % n)
                        continue
                    samples[n].append((c1 - c0) / work)
                    walls[n].append(res["wall"] / work)
        metric = walls if a.metric == "wall" else samples
        med = {n: statistics.median(metric[n]) for n in names if metric[n]}
        row = {"id": e.id, "threads": thr or e.threads, "med": med, "n": {n: len(metric[n]) for n in names}, "bad": bad,
               "spread": {n: (max(metric[n]) - min(metric[n])) / med[n] for n in med if med[n] > 0}}
        row["ratio"] = {n: med[ref] / med[n] for n in med if n != ref and ref in med and med[n] > 0}
        rows.append(row)
        print("ab: %-40s %s %s" % (e.id, " ".join("%s=%.3f" % (n, med.get(n, float("nan"))) for n in names),
                                    ("BAD " + "; ".join(bad)) if bad else ""), flush=True)
        if a.json:
            json.dump({"metric": a.metric, "ref": ref, "builds": builds, "rows": rows}, open(a.json, "w"), indent=1)
    print()
    print("| program | threads | %s | %s |" % (" | ".join("%s s" % n for n in names), " | ".join("%s/%s" % (ref, n) for n in names if n != ref)))
    print("|---|---:|%s|%s|" % ("---:|" * len(names), "---:|" * (len(names) - 1)))
    for r in rows:
        print("| %s | %d | %s | %s |" % (r["id"], r["threads"], " | ".join("%.3f" % r["med"].get(n, float("nan")) for n in names),
                                          " | ".join("%.3fx" % r["ratio"].get(n, float("nan")) for n in names if n != ref)))
    print()
    groups = sorted(set(r["id"].split("/")[0] for r in rows)) + ["all"]
    for g in groups:
        sel = [r for r in rows if g == "all" or r["id"].split("/")[0] == g]
        parts = []
        for n in names:
            if n == ref:
                continue
            xs = [r["ratio"][n] for r in sel if n in r["ratio"] and not r["bad"]]
            if xs:
                parts.append("%s/%s geomean %.3fx (min %.3fx, max %.3fx, %d programs)" % (ref, n, geomean(xs), min(xs), max(xs), len(xs)))
        print("group %-6s %s" % (g, "; ".join(parts)))
    return 1 if any(r["bad"] for r in rows) else 0


def cmd_bits(a):
    builds = parse_builds(a.builds)
    names = list(builds)
    ref = a.ref or names[0]
    ctl = [n for n in names if n != ref and n.startswith(ref)]
    rn = Runner(a)
    threads = [int(x) for x in a.threads_list.split(",")]
    worst = 0
    rows = []
    for e in select(a):
        for t in threads:
            res = {}
            for n in names:
                res[n] = rn.run(e, builds[n], a.cpus if t == 1 else a.threads_cpus, t, collect=True, timeout=e.cap * 2)
            keys = ("rc", "out", "digest", "counters", "final")
            diffs = {}
            for n in names:
                if n == ref:
                    continue
                diffs[n] = [k for k in keys if res[n][k] != res[ref][k]]
            nondet = [n for n in ctl if diffs[n]]
            pgo = [n for n in names if n != ref and n not in ctl]
            status = "IDENTICAL"
            if any(diffs[n] for n in pgo):
                # differs from REF: is it also different between REF and its own copy?
                status = "NONDETERMINISTIC (REF differs from its copy: %s)" % ",".join(sorted(set(k for n in nondet for k in diffs[n]))) if nondet else \
                    "DIFFERENT: " + "; ".join("%s: %s" % (n, ",".join(diffs[n])) for n in pgo if diffs[n])
                if not nondet:
                    worst = 1
            elif any(res[n]["rc"] != 0 for n in names):
                status = "IDENTICAL (rc %s)" % ",".join(str(res[n]["rc"]) for n in names)
            rows.append({"id": e.id, "threads": t, "status": status, "digest": res[ref]["digest"], "counters": res[ref]["counters"],
                         "final": res[ref]["final"], "out": res[ref]["out"], "rc": res[ref]["rc"]})
            print("bits: %-40s t=%d %s" % (e.id, t, status), flush=True)
            if a.json:
                json.dump(rows, open(a.json, "w"), indent=1)
    n_ident = sum(1 for r in rows if r["status"].startswith("IDENTICAL"))
    n_nondet = sum(1 for r in rows if r["status"].startswith("NONDET"))
    n_diff = sum(1 for r in rows if r["status"].startswith("DIFFERENT"))
    print("bits: %d runs: %d identical, %d nondeterministic on the plain build itself, %d DIFFERENT" % (len(rows), n_ident, n_nondet, n_diff))
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["time", "bits"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--builds", required=True, help="NAME=DIR[,NAME=DIR...]; the first (or --ref) is the reference")
    ap.add_argument("--ref", default="")
    ap.add_argument("--corp", default=os.environ.get("PGO_CORP", ""))
    ap.add_argument("--py", default=os.environ.get("PGO_PY", sys.executable))
    ap.add_argument("--src", default=os.path.dirname(os.path.dirname(HERE)))
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--runs", type=int, default=9)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--cpus", default="9", help="CPU for one-thread runs")
    ap.add_argument("--threads", type=int, default=1, help="time: override VGPU_THREADS for every entry (>1: wall time on --threads-cpus)")
    ap.add_argument("--threads-cpus", default="0-7")
    ap.add_argument("--threads-list", default="1,4", help="bits: thread counts")
    ap.add_argument("--metric", choices=["cpu", "wall"], default="cpu")
    ap.add_argument("--lock", default="")
    ap.add_argument("--max-load", type=float, default=8.0)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    if a.mode == "time" and a.threads > 1 and a.metric == "cpu":
        a.metric = "wall"
    return cmd_time(a) if a.mode == "time" else cmd_bits(a)


if __name__ == "__main__":
    sys.exit(main())
