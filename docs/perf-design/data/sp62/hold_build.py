#!/usr/bin/env python3
"""hold_build.py <build dir> <target>...: builds the targets' objects a few at a time, each group to its end, under the machine lock
in holds of about --hold seconds (a group that starts inside a hold finishes in it), then links. Nothing is killed at a hold's end."""
import fcntl, os, re, subprocess, sys, time
L = "/tmp/pantheonsim-builder.lock"
HOLD = float(os.environ.get("HOLD", "200"))
b, targets = sys.argv[1], sys.argv[2:]
cmds = subprocess.run(["ninja", "-C", b, "-t", "commands"] + targets, capture_output=True, text=True, check=True).stdout.splitlines()
objs = []
for c in cmds:
    m = re.search(r" -o (\S+\.o)\b", c)
    if m and m.group(1) not in objs: objs.append(m.group(1))
# what is out of date
dry = subprocess.run(["ninja", "-C", b, "-n", "-d", "explain"] + targets, capture_output=True, text=True).stdout
todo = [o for o in objs if ("Building CXX object " + o) in dry or ("Building C object " + o) in dry]
print("objects %d, to build %d" % (len(objs), len(todo)), flush=True)
lock = open(L, "w")
i = 0
while True:
    fcntl.flock(lock, fcntl.LOCK_EX)
    t0 = time.monotonic()
    while i < len(todo) and time.monotonic() - t0 < HOLD:
        g = todo[i:i + 3]
        r = subprocess.run(["nice", "-n", "19", "ninja", "-C", b, "-j3"] + g, capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-3000:], flush=True); sys.exit(1)
        i += len(g)
    if i >= len(todo) and time.monotonic() - t0 < HOLD:
        r = subprocess.run(["nice", "-n", "19", "ninja", "-C", b, "-j3"] + targets, capture_output=True, text=True)
        fcntl.flock(lock, fcntl.LOCK_UN)
        print(r.stdout[-1500:], flush=True)
        sys.exit(r.returncode)
    fcntl.flock(lock, fcntl.LOCK_UN)
    print("built %d/%d, hold %.0f s" % (i, len(todo), time.monotonic() - t0), flush=True)
    time.sleep(2)
