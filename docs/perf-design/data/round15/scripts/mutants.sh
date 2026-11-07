#!/bin/bash
# mutants.sh <builddir-name> <program.arch> <logname> <mutant...>
# For each mutant: the JIT (every region compiled) with VGPU_JIT_MUTANT=<mutant> against the tier, on a prebuilt e2e program; 3 at a time, inside one hold of the machine lock.
b=$1; prog=$2; log=$3; shift 3
W=/tmp/sp33-jit/e2ework; L=/tmp/sp33-jit/work/$log
export B=/tmp/sp33-jit/$b/build prog W L
cat > /tmp/sp33-jit/work/mutant_one.sh <<'EOS'
#!/bin/bash
m=$1
base=(VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 VGPU_THREADS=1 VGPU_COUNTERS=1 VGPU_SASS_LOG=1 LD_LIBRARY_PATH=$B/shim)
jit=(VGPU_TIER=1 VGPU_JIT=1 VGPU_JIT_THRESHOLD=1 VGPU_JIT_SYNC=1 VGPU_JIT_HOT_PERMILLE=0 VGPU_JIT_MAX_OPS=100000000 VGPU_JIT_MIN_REGION=2)
cache=/tmp/sp33-jit/work/mc-$prog-$m; rm -rf $cache; mkdir -p -m 700 $cache
ref=$(cd $W && env "${base[@]}" VGPU_TIER=1 timeout 900 ./$prog 2>&1)
if [ "$m" = none ]; then out=$(cd $W && env "${base[@]}" "${jit[@]}" VGPU_JIT_CACHE=$cache timeout 1800 ./$prog 2>&1)
else out=$(cd $W && env "${base[@]}" "${jit[@]}" VGPU_JIT_MUTANT=$m VGPU_JIT_CACHE=$cache timeout 1800 ./$prog 2>&1); fi
if [ "$out" == "$ref" ]; then echo "$prog $m: SURVIVED (output identical to the tier)"; else echo "$prog $m: caught ($(diff <(echo "$ref") <(echo "$out") | grep -c '^>') lines differ)"; fi
rm -rf $cache
EOS
chmod +x /tmp/sp33-jit/work/mutant_one.sh
printf '%s\n' "$@" | flock /tmp/pantheonsim-builder.lock nice -n 19 xargs -P3 -I{} /tmp/sp33-jit/work/mutant_one.sh {} >> $L 2>&1
echo "=== done $(date +%T)" >> $L
