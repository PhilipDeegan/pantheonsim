#!/usr/bin/env python3
"""CI's memory reservation and the setup action's build cache stay wired (#575, #576, #582).

Text checks on the workflow files, the standard library only (nothing here runs a runner):

  1. Every job that calls `ci/jobs.sh --check` (it takes a share of the host's ledger) also has a
     step `ci/jobs.sh --release` with `if: always()` after it, so a failed or cancelled job gives
     the share back. A job that opts out says `CI_RESERVE: '0'` on its check step.
  2. Every such job that runs in a container on a pool (its options name BURST_CONTAINER_OPTIONS)
     bind-mounts the ledger directory, or its share would be taken in a private /tmp that no other
     runner sees.
  3. The setup action's build cache is restore plus save, the save right after the build, only on
     a miss, both pinned by commit SHA with the version comment; there is no combined
     actions/cache step whose post step runs only when the whole job succeeds.

usage: check_ci_reservation.py [--mutate-self-test]
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
LEDGER = "/tmp/pantheonsim-ci-reserve"


def jobs(text):
    """{job name: body} for the top-level jobs of a workflow."""
    m = re.search(r"^jobs:\s*$", text, re.M)
    if not m:
        return {}
    body = text[m.end():]
    parts = re.split(r"^  ([A-Za-z0-9_-]+):\s*$", body, flags=re.M)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def steps(body, indent=6):
    """List of step texts (a workflow's steps sit at 6 spaces, a composite action's at 4)."""
    return re.split(r"^" + " " * indent + r"- ", body, flags=re.M)[1:]


def check_workflows(files):
    errs = []
    for path, text in files:
        for name, body in jobs(text).items():
            sl = steps(body)
            idx = [i for i, s in enumerate(sl) if re.search(r"ci/jobs\.sh --check \S", s)]
            if not idx:
                continue
            where = f"{path}: job {name}"
            if any("CI_RESERVE: '0'" in sl[i] for i in idx):
                continue
            rel = [i for i, s in enumerate(sl) if re.search(r"ci/jobs\.sh --release\s*$", s, re.M)]
            if not rel:
                errs.append(f"{where}: takes a reservation (ci/jobs.sh --check) and never runs ci/jobs.sh --release")
            else:
                if not any(i > idx[0] and re.search(r"if:\s*always\(\)", sl[i]) for i in rel):
                    errs.append(f"{where}: ci/jobs.sh --release is not `if: always()` after the check")
            if "BURST_CONTAINER_OPTIONS" in body and f"-v {LEDGER}:{LEDGER}" not in body:
                errs.append(f"{where}: the container does not bind-mount {LEDGER}")
    return errs


def check_action(text):
    errs = []
    if re.search(r"uses:\s*actions/cache@", text):
        errs.append("action.yml: a combined actions/cache step saves only when the whole job succeeds")
    pins = re.findall(r"uses:\s*actions/cache/(restore|save)@(\S+)(.*)", text)
    kinds = sorted(k for k, _, _ in pins)
    if kinds != ["restore", "save"]:
        errs.append(f"action.yml: wants one actions/cache/restore and one actions/cache/save, has {kinds}")
    for k, ref, rest in pins:
        if not re.fullmatch(r"[0-9a-f]{40}", ref) or not re.search(r"#\s*v\d", rest):
            errs.append(f"action.yml: actions/cache/{k} is not pinned by commit SHA with a version comment")
    sl = steps(text, 4)
    b = [i for i, s in enumerate(sl) if "Build the simulator" in s.split("\n")[0]]
    sv = [i for i, s in enumerate(sl) if "actions/cache/save@" in s]
    if b and sv:
        if sv[0] != b[0] + 1:
            errs.append("action.yml: the save step is not right after the build step")
        if "steps.cache.outputs.cache-hit != 'true'" not in sl[sv[0]]:
            errs.append("action.yml: the save step is not guarded by cache-hit != 'true'")
    else:
        errs.append("action.yml: no build step or no save step")
    return errs


def main():
    wf = [(p.relative_to(ROOT).as_posix(), p.read_text()) for p in sorted((ROOT / ".github/workflows").glob("*.yml"))]
    action = (ROOT / "action.yml").read_text()
    errs = check_workflows(wf) + check_action(action)

    # The checks must be able to fail: each mutation below has to be caught.
    ci = dict(wf)[".github/workflows/ci.yml"]
    for label, text in [
        ("a missing release step", ci.replace("ci/jobs.sh --release", "true")),
        ("a release that is not always()", ci.replace("if: always()\n        run: ci/jobs.sh --release", "run: ci/jobs.sh --release")),
        ("a missing mount", ci.replace(f"-v {LEDGER}:{LEDGER} ", "")),
    ]:
        if not check_workflows([(".github/workflows/ci.yml", text)]):
            errs.append(f"self-test: {label} was not caught")
    for label, text in [
        ("a combined cache step", action.replace("actions/cache/restore@", "actions/cache@")),
        ("a save with no guard", action.replace("if: steps.cache.outputs.cache-hit != 'true'\n      uses: actions/cache/save", "uses: actions/cache/save")),
        ("an unpinned save", re.sub(r"(actions/cache/save@)[0-9a-f]{40}", r"\1v4", action)),
    ]:
        if not check_action(text):
            errs.append(f"self-test: {label} was not caught")

    for e in errs:
        print("FAIL:", e)
    if errs:
        return 1
    print("PASS: every reservation is released, mounted where needed, and the setup action saves its build cache right after the build")
    return 0


sys.exit(main())
