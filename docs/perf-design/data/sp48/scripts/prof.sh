#!/bin/bash
# prof.sh <out-prefix> <build> <mode: tier|jit> <cwd> <argv...>
# One IBS-op profile of one program on the simulator (server3, AMD Zen 4): `ibs_op/cnt_ctl=1/` samples every N dispatched micro-ops with the
# exact instruction address, so the shares are shares of executed work (not of stalled time) and need no skid correction. A second run with
# cnt_ctl=0 gives the same split by cycles. Holds the machine lock for the runs. JIT mode uses a private cache, filled by an untimed first run.
set -e
out=$1; build=$2; mode=$3; cwd=$4; shift 4
cpu=${CPU:-6}
envs="VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 VGPU_THREADS=1 LD_LIBRARY_PATH=$build/shim"
if [ "$mode" = jit ]; then
  mkdir -p "$out.cache"; chmod 700 "$out.cache"   # (the JIT only trusts a cache directory that is the user's alone)
  envs="$envs VGPU_JIT=1 VGPU_JIT_THRESHOLD=100000 VGPU_JIT_SYNC=1 VGPU_JIT_CACHE=$out.cache VGPU_JIT_STATS=1"
fi
cd "$cwd"
# (the caller holds the machine lock when NOLOCK=1: a batch of profiles in one hold)
if [ -z "$NOLOCK" ]; then exec 9>>/tmp/pantheonsim-builder.lock; flock 9; fi
[ "$mode" = jit ] && env $envs taskset -c $cpu "$@" > "$out.warm.log" 2>&1 || true
for ctl in 1 0; do
  sudo perf record -q -e ibs_op/cnt_ctl=$ctl/ -c 400009 -o "$out.ctl$ctl.data" -- sudo -u "$USER" env $envs nice -n 19 taskset -c $cpu "$@" > "$out.ctl$ctl.log" 2>&1 || echo "rc=$?" >> "$out.ctl$ctl.log"
  sudo chown "$USER" "$out.ctl$ctl.data"
  perf script -i "$out.ctl$ctl.data" --no-demangle -F comm,ip,sym,symoff,dso,dsoff 2>/dev/null > "$out.ctl$ctl.script" || true
done
sudo perf stat -x, -e instructions:u,cycles:u -o "$out.stat" -- sudo -u "$USER" env $envs nice -n 19 taskset -c $cpu "$@" > /dev/null 2>&1 || true
sudo chown "$USER" "$out.stat" 2>/dev/null || true
[ -z "$NOLOCK" ] && flock -u 9
true
