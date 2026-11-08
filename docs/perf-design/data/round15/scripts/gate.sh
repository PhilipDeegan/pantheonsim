#!/bin/bash
# gate.sh <builddir-name> <srcdir-name> <logname> [shards...]: the e2e_sass_jit gate stages one at a time, each under the machine lock, against prebuilt programs
b=$1; s=$2; log=$3; shift 3
shards=${@:-tier steps steps_gather steps_shared modes}
for sh in $shards; do
  echo "=== shard $sh $(date +%T)" >> /tmp/sp33-jit/work/$log
  VGPU_BUILD_DIR=/tmp/sp33-jit/$b/build VGPU_TIER_WORK=/tmp/sp33-jit/e2ework VGPU_JIT_SHARD=$sh \
    flock /tmp/pantheonsim-builder.lock nice -n 19 taskset -c 12-14 /tmp/sp33-jit/$s/src/nvidia/tests/e2e/run_sass_jit.sh >> /tmp/sp33-jit/work/$log 2>&1
  echo "=== shard $sh rc=$? $(date +%T)" >> /tmp/sp33-jit/work/$log
done
echo "=== all done" >> /tmp/sp33-jit/work/$log
