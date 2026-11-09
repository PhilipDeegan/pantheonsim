#!/usr/bin/env python3
"""Run the programs of a PGO manifest against an instrumented build: the training step of scripts/pgo-build.sh.

  train.py --manifest scripts/pgo/train-manifest.txt --build build-pgo --corp /path/to/corpus --py /path/to/python
           [--jobs 2] [--only REGEX] [--skip REGEX] [--min-ok 0.9] [--log DIR] [--json FILE]
  train.py ... --check      only validate the manifest and report which programs are absent
  train.py ... --dry-run    print the command of every run

Every entry runs `reps` times with a wall-clock cap. The instrumented libraries append to the same .gcda files
(libgcov locks and merges them), so entries may run side by side (--jobs). Exit status: 0 when at least --min-ok of the
entries finished (rc 0, inside the cap), 1 otherwise, 2 for a bad manifest or build. A run that fails, is capped or
whose program is missing is listed; nothing is silently skipped.
"""
import argparse
import concurrent.futures
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import manifest as mf  # noqa: E402


def run_once(entry, ctx, scale, lock, nice, logdir, env_base=None):
    """-> (status, seconds, rc); status in ok, failed, capped, missing."""
    ctx = dict(ctx)
    scratch = tempfile.mkdtemp(prefix="pgo-train-")
    ctx["TMP"] = scratch
    argv = [mf.subst(a, ctx) for a in entry.command]
    exe = argv[0]
    if not (os.path.isfile(exe) and os.access(exe, os.X_OK)) and shutil.which(exe) is None:
        shutil.rmtree(scratch, ignore_errors=True)
        return "missing", 0.0, None
    env = mf.make_env(entry, ctx, env_base)
    cwd = scratch if entry.cwd == "tmp" else mf.subst(entry.cwd, ctx)
    if nice:
        argv = ["nice", "-n", "19"] + argv
    cap = entry.cap * scale
    lf = None
    if lock:
        import fcntl
        lf = open(lock, "a+")
        fcntl.flock(lf, fcntl.LOCK_EX)
    out = b""
    status, rc = "ok", None
    t0 = time.monotonic()
    try:
        p = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            out, _ = p.communicate(timeout=cap)
            rc = p.returncode
            status = "ok" if rc == 0 else "failed"
        except subprocess.TimeoutExpired:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(p.pid, sig)
                except OSError:
                    pass
                try:
                    out, _ = p.communicate(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    continue
            rc = p.returncode
            status = "capped"
    finally:
        sec = time.monotonic() - t0
        if lf:
            import fcntl
            fcntl.flock(lf, fcntl.LOCK_UN)
            lf.close()
        shutil.rmtree(scratch, ignore_errors=True)
    if logdir and status != "ok":
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, entry.id.replace("/", "_") + ".log"), "wb") as f:
            f.write(out[-20000:])
    return status, sec, rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--build", required=True, help="the instrumented build directory (has shim/ and vgpu)")
    ap.add_argument("--corp", default=os.environ.get("PGO_CORP", ""), help="corpus directory (@CORP@)")
    ap.add_argument("--py", default=os.environ.get("PGO_PY", sys.executable), help="python with PyTorch (@PY@)")
    ap.add_argument("--src", default=os.path.dirname(os.path.dirname(HERE)), help="source tree (@SRC@)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--min-ok", type=float, default=0.9, help="fraction of entries that must finish (default 0.9)")
    ap.add_argument("--cap-scale", type=float, default=1.0, help="multiply every cap (slow machine)")
    ap.add_argument("--lock", default="", help="flock file taken around every run")
    ap.add_argument("--no-nice", action="store_true")
    ap.add_argument("--log", default="", help="directory for the output of runs that did not finish")
    ap.add_argument("--json", default="")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        entries = mf.parse(a.manifest)
    except (OSError, ValueError) as e:
        print("train: bad manifest: %s" % e, file=sys.stderr)
        return 2
    if a.only:
        entries = [e for e in entries if re.search(a.only, e.id)]
    if a.skip:
        entries = [e for e in entries if not re.search(a.skip, e.id)]
    if not entries:
        print("train: no entries selected", file=sys.stderr)
        return 2
    build = os.path.abspath(a.build)
    if not os.path.isdir(os.path.join(build, "shim")) and not a.check:
        print("train: %s/shim not found: build the instrumented tree first" % build, file=sys.stderr)
        return 2
    ctx = {"CORP": os.path.abspath(a.corp) if a.corp else "", "BUILD": build, "SRC": os.path.abspath(a.src), "PY": a.py}
    if a.check or a.dry_run:
        missing = 0
        for e in entries:
            c = dict(ctx, TMP="/tmp/pgo-train-XXXX")
            argv_ = [mf.subst(x, c) for x in e.command]
            have = os.path.isfile(argv_[0]) or shutil.which(argv_[0])
            missing += 0 if have else 1
            if a.dry_run:
                print("%s x%d cap %gs threads %d kind %s: %s" % (e.id, e.reps, e.cap, e.threads, e.kind, " ".join(argv_)))
            else:
                print("%-44s %s" % (e.id, "present" if have else "MISSING " + argv_[0]))
        print("train: %d entries, %d runs, %d missing" % (len(entries), sum(e.reps for e in entries), missing))
        return 0
    jobs = [(e, r) for e in entries for r in range(e.reps)]
    results = []
    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        futs = {ex.submit(run_once, e, ctx, a.cap_scale, a.lock or None, not a.no_nice, a.log or None): (e, r)
                for e, r in jobs}
        for f in concurrent.futures.as_completed(futs):
            e, r = futs[f]
            status, sec, rc = f.result()
            results.append({"id": e.id, "rep": r, "status": status, "seconds": round(sec, 2), "rc": rc})
            print("train: %-44s rep %d %-7s %7.1fs%s" % (e.id, r + 1, status, sec, "" if rc in (0, None) else " rc=%s" % rc), flush=True)
    by = {}
    for r in results:
        by.setdefault(r["id"], []).append(r["status"])
    ok = [i for i, s in by.items() if all(x == "ok" for x in s)]
    bad = {i: s for i, s in by.items() if i not in ok}
    total = time.monotonic() - t0
    print("train: %d of %d entries finished (%d runs, %.0f s wall)" % (len(ok), len(by), len(results), total))
    for i, s in sorted(bad.items()):
        print("train: NOT FINISHED %s: %s" % (i, ",".join(sorted(set(x for x in s if x != "ok")))), file=sys.stderr)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"wall_s": round(total, 1), "runs": results}, f, indent=1)
    return 0 if len(ok) >= a.min_ok * len(by) else 1


if __name__ == "__main__":
    sys.exit(main())
