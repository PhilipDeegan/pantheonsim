#!/bin/bash
# perfprof.sh <label> <build> <cachedir> <program path>: perf record of one warm JIT run (under the machine lock), the top symbols
label=$1; b=$2; cache=$3; prog=$4
cd $(dirname $prog)
flock /tmp/pantheonsim-builder.lock nice -n 19 taskset -c 9 env VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 VGPU_THREADS=1 LD_LIBRARY_PATH=/tmp/sp33-jit/$b/build/shim VGPU_JIT=1 VGPU_JIT_THRESHOLD=100000 VGPU_JIT_SYNC=1 VGPU_JIT_CACHE=$cache perf record -F 1500 -o /tmp/sp33-jit/work/perf.$label.data $prog > /dev/null 2>&1
perf report -i /tmp/sp33-jit/work/perf.$label.data --stdio --sort dso,sym 2>/dev/null | grep -v "^#" | grep . | head -40 > /tmp/sp33-jit/work/perf.$label.txt
rm -f /tmp/sp33-jit/work/perf.$label.data
