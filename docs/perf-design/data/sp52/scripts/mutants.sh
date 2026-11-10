#!/bin/bash
# mutants.sh <build with -DVGPU_JIT_MUTANTS=ON> [work dir]: the linking tests on the pristine build and under each mutant. Two tests:
#   unit  test_jit_resid "linked" (the linked code against the unlinked code and the run loop's rules between its calls);
#   sass  test_sass_chain_cont with the JIT forced on and VGPU_JIT_LINK=1 (hand-written SASS kernels against the interpreter).
# A generator mutant (link_*) must fail the unit test; a host mutant (host_*) must fail the sass test; the pristine build passes both.
# Holds the machine lock (NOLOCK=1: the caller does); one compiler at a time.
b=$1; w=${2:-/tmp}
if [ -z "$NOLOCK" ]; then exec 9>>/tmp/pantheonsim-builder.lock; flock 9; fi
J="VGPU_JIT=1 VGPU_JIT_LINK=1 VGPU_JIT_STATS=1 VGPU_JIT_STATS_FILE=/dev/null VGPU_JIT_THRESHOLD=1 VGPU_JIT_SYNC=1 VGPU_JIT_HOT_PERMILLE=0 VGPU_JIT_MIN_REGION=2 VGPU_JIT_MAX_OPS=100000000 VGPU_QUIET=1"
run() {
  m=$1
  VGPU_JIT_MUTANT=$m TMPDIR=$w nice -n 19 "$b/test_jit_resid" linked > "$w/sp52-mut.$$" 2>&1; rc1=$?
  u=$(grep -E "link:|FAIL|differs" "$w/sp52-mut.$$" | head -2 | tr '\n' ' ' | cut -c1-330)
  rm -rf "$w/sp52-jc.$$"; mkdir -m 700 "$w/sp52-jc.$$"
  env $J VGPU_JIT_MUTANT=$m VGPU_JIT_CACHE="$w/sp52-jc.$$" nice -n 19 "$b/test_sass_chain_cont" > "$w/sp52-mut.$$" 2>&1; rc2=$?
  s=$(grep -E "linked inside|FAIL|with the tier and" "$w/sp52-mut.$$" | head -2 | tr '\n' ' ' | cut -c1-260)
  rm -rf "$w/sp52-jc.$$" "$w/sp52-mut.$$"
  echo "mutant=${m:-none} unit rc=$rc1 sass rc=$rc2 | unit: $u | sass: $s"
}
run ""
for m in link_flag link_olo link_olo_eq link_spin link_spin_limit link_nosplit link_split_pcs link_split_olo link_split_fl link_split_group link_split_glanes link_bssy link_bssy_ex link_exit_go link_exit_cb link_call_cb host_olo host_spins; do run $m; done
[ -z "$NOLOCK" ] && flock -u 9
true
