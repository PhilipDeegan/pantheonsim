#!/bin/bash
# unit_mutants.sh <builddir-name> <logname> <test binary> <mutant...>: the unit test under each mutant, one after the other on 3 CPUs (a mutant that makes a loop endless is killed after 10 minutes: caught as a hang), one hold of the machine lock
b=$1; log=$2; t=$3; shift 3
export B=/tmp/sp33-jit/$b/build T=$t L=/tmp/sp33-jit/work/$log
cat > /tmp/sp33-jit/work/unitmut_one.sh <<'EOS'
#!/bin/bash
m=$1
cd $B
if [ "$m" = none ]; then out=$(VGPU_JIT_GRAPH_UNITS=3 VGPU_JIT_RANDOM_UNITS=3 timeout 600 taskset -c 12-14 ./$T 2>&1); else out=$(VGPU_JIT_MUTANT=$m VGPU_JIT_GRAPH_UNITS=3 VGPU_JIT_RANDOM_UNITS=3 timeout 600 taskset -c 12-14 ./$T 2>&1); fi
rc=$?
fails=$(echo "$out" | grep -c "^\[ FAIL \]")
first=$(echo "$out" | grep -m1 "^\[ FAIL \]" | cut -c1-110)
[ $rc = 124 ] && first="HANG (killed after 600 s)"
echo "$T $m: rc=$rc failed_tests=$fails $(echo "$out" | grep -E "tests passed" | head -1) $first"
EOS
chmod +x /tmp/sp33-jit/work/unitmut_one.sh
printf '%s\n' "$@" | flock /tmp/pantheonsim-builder.lock nice -n 19 xargs -P1 -I{} /tmp/sp33-jit/work/unitmut_one.sh {} >> $L 2>&1
echo "=== done $(date +%T)" >> $L
