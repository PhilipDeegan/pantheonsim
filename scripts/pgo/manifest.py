"""Reader for the PGO program manifests (scripts/pgo/train-manifest.txt, test-manifest.txt). Stdlib only.

One program per line, eight fields separated by '|', '#' starts a comment:

  id | kind | threads | reps | cap_s | cwd | env | command

  id       unique name, "<group>/<corpus>/<program>" (the group before the first '/' is nv, amd, pan, torch or tiny)
  kind     sass  NVIDIA program, SASS engine, VGPU_GPU=nvidia/rtx3060
           ptx   NVIDIA program, PTX interpreter (VGPU_SASS=0)
           amd   AMD program, VGPU_GPU=amd/mi300x
           raw   nothing is set for you (the env field and the command carry everything)
  threads  VGPU_THREADS for the run
  reps     how many times the training run executes the program (the weighting: programs that execute few
           instructions are repeated so that no one program dominates the profile); ignored by the A/B tools
  cap_s    wall-clock limit of one run in seconds; a run that exceeds it is killed and counts as CAPPED
           (an instrumented process killed from outside writes no profile data)
  cwd      "tmp" (a fresh empty directory per run, removed afterwards) or a directory
  env      space-separated KEY=VALUE pairs, or "-"
  command  the program and its arguments (split like a shell, no pipes or redirections)

Placeholders in cwd, env and command: @CORP@ (the corpus directory), @BUILD@ (the build directory under test), @SRC@
(the source tree), @PY@ (python with PyTorch), @TMP@ (the run's scratch directory).
"""
import os
import re
import shlex

KINDS = ("sass", "ptx", "amd", "raw")


class Entry:
    __slots__ = ("id", "kind", "threads", "reps", "cap", "cwd", "env", "command", "line")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    @property
    def group(self):
        return self.id.split("/", 1)[0]


def parse(path):
    entries, seen = [], set()
    with open(path) as f:
        for n, raw in enumerate(f, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("|")]
            where = "%s:%d" % (path, n)
            if len(parts) != 8:
                raise ValueError("%s: %d fields, want 8 (id|kind|threads|reps|cap_s|cwd|env|command)" % (where, len(parts)))
            i, kind, thr, reps, cap, cwd, env, cmd = parts
            if not re.fullmatch(r"[A-Za-z0-9_.+-]+(/[A-Za-z0-9_.+-]+)+", i):
                raise ValueError("%s: bad id %r" % (where, i))
            if i in seen:
                raise ValueError("%s: duplicate id %s" % (where, i))
            seen.add(i)
            if kind not in KINDS:
                raise ValueError("%s: kind %r not in %s" % (where, kind, KINDS))
            try:
                thr, reps, cap = int(thr), int(reps), float(cap)
            except ValueError:
                raise ValueError("%s: threads, reps and cap_s must be numbers" % where)
            if thr < 1 or reps < 1 or cap <= 0:
                raise ValueError("%s: threads and reps are >= 1 and cap_s > 0" % where)
            envd = {}
            if env != "-":
                for kv in env.split():
                    if "=" not in kv:
                        raise ValueError("%s: env item %r is not KEY=VALUE" % (where, kv))
                    k, v = kv.split("=", 1)
                    envd[k] = v
            argv = shlex.split(cmd)
            if not argv:
                raise ValueError("%s: empty command" % where)
            entries.append(Entry(id=i, kind=kind, threads=thr, reps=reps, cap=cap, cwd=cwd, env=envd, command=argv, line=n))
    return entries


def subst(s, ctx):
    for k, v in ctx.items():
        s = s.replace("@%s@" % k, v)
    return s


def make_env(entry, ctx, base=None):
    """The environment of one run: the parent's without VGPU_*, then the kind's settings, then the entry's."""
    env = {k: v for k, v in (base if base is not None else os.environ).items() if not k.startswith("VGPU_")}
    env["VGPU_QUIET"] = "1"
    env["VGPU_THREADS"] = str(entry.threads)
    if ctx.get("BUILD") and entry.kind != "raw":
        env["LD_LIBRARY_PATH"] = os.path.join(ctx["BUILD"], "shim")
    if entry.kind in ("sass", "ptx"):
        env["VGPU_GPU"] = "nvidia/rtx3060"
        if entry.kind == "ptx":
            env["VGPU_SASS"] = "0"
    elif entry.kind == "amd":
        env["VGPU_GPU"] = "amd/mi300x"
    for k, v in entry.env.items():
        env[k] = subst(v, ctx)
    return env


def program_path(entry, ctx):
    """The executable of an entry after substitution (for the existence check)."""
    return subst(entry.command[0], ctx)
