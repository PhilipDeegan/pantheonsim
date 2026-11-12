#!/bin/bash
# mutants.sh <build> <work>: round sp57's two tests on the pristine build and under each mutant of the wait on the plain loop (needs -DVGPU_JIT_MUTANTS=ON).
# unit = test_jit_wait; sass = test_sass_chain_cont with the environment of ctest's test_sass_chain_cont_jit_wait. rc 0: the test passed.
b=$1; w=$2; mkdir -p "$w"
J="VGPU_JIT=1 VGPU_JIT_EXPECT_WAIT=1 VGPU_JIT_THRESHOLD=20000 VGPU_JIT_LOOKUP_AFTER=256 VGPU_JIT_PROFILE=2000 VGPU_JIT_SYNC=1 VGPU_JIT_HOT_PERMILLE=0 VGPU_JIT_MIN_REGION=2 VGPU_JIT_MAX_OPS=100000000 VGPU_QUIET=1"
for m in none wait_never wait_stuck wait_no_credit wait_no_profile; do
  [ "$m" = none ] && mm= || mm=$m
  VGPU_JIT_MUTANT=$mm TMPDIR=$w "$b/test_jit_wait" > "$w/o1" 2>&1; rc1=$?
  rm -rf "$w/jc"; mkdir -m 700 "$w/jc"
  env $J VGPU_JIT_MUTANT=$mm VGPU_JIT_CACHE="$w/jc" "$b/test_sass_chain_cont" > "$w/o2" 2>&1; rc2=$?
  echo "mutant=$m unit rc=$rc1 sass rc=$rc2 | $(grep -h -m1 -E 'VCHECK[_A-Z]*\(' "$w/o1" | cut -c1-110) | $(grep -h -E 'sections that waited|VCHECK[_A-Z]*\(' "$w/o2" | head -2 | tr '\n' ' ' | cut -c1-200)"
done
rm -rf "$w/jc" "$w/o1" "$w/o2"
