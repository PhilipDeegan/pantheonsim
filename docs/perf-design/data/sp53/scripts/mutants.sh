#!/bin/bash
# NOTE: VGPU_JIT_LINK_BSYNC (BSYNC as a region's op) was removed from the code after round sp53; this script is kept as the record of that round and no longer runs as written.
# mutants.sh <build with -DVGPU_JIT_MUTANTS=ON> [work dir]: round sp53's tests on the pristine build and under each of the round's mutants. Four tests:
#   unit  test_jit_resid "linked" (the linked code, BSYNCs included, against the unlinked code and the run loop's rules between its calls);
#   rows  test_jit_resid "rows"   (the object of a kernel of SYR2K's shape holds no string instruction and no out-of-line escape);
#   sass  test_sass_chain_cont with the JIT forced on, VGPU_JIT_LINK=1 and VGPU_JIT_LINK_RULE=0 (hand-written SASS kernels against the interpreter; BSYNCs
#         must have been linked): ctest's test_sass_chain_cont_jit_link;
#   rule  the same with the loop rule on (what it declines, what it leaves linked): ctest's test_sass_chain_cont_jit_link_rule.
# A generator mutant (link_bsync_*) must fail the unit test, row_loop and escape_outline the rows test, a host mutant (host_bsync_*) the sass test, link_rule_*
# the rule test; the pristine build passes all four. ONLY='a b' runs those mutants only. Holds the machine lock (NOLOCK=1: the caller does).
b=$1; w=${2:-/tmp}
if [ -z "$NOLOCK" ]; then exec 9>>/tmp/pantheonsim-builder.lock; flock 9; fi
J="VGPU_JIT=1 VGPU_JIT_LINK=1 VGPU_JIT_LINK_BSYNC=1 VGPU_JIT_STATS=1 VGPU_JIT_STATS_FILE=/dev/null VGPU_JIT_THRESHOLD=1 VGPU_JIT_SYNC=1 VGPU_JIT_HOT_PERMILLE=0 VGPU_JIT_MIN_REGION=2 VGPU_JIT_MAX_OPS=100000000 VGPU_QUIET=1"
run() {
  m=$1
  VGPU_JIT_MUTANT=$m TMPDIR=$w nice -n 19 "$b/test_jit_resid" linked > "$w/sp53-mut.$$" 2>&1; rc1=$?
  u=$(grep -E "FAIL|differs" "$w/sp53-mut.$$" | head -1 | cut -c1-120)
  VGPU_JIT_MUTANT=$m TMPDIR=$w nice -n 19 "$b/test_jit_resid" rows > "$w/sp53-mut.$$" 2>&1; rc3=$?
  r=$(grep -E "rows:" "$w/sp53-mut.$$" | sed -E 's/^ +rows: the object //' | tr '\n' ';' | cut -c1-220)
  rm -rf "$w/sp53-jc.$$"; mkdir -m 700 "$w/sp53-jc.$$"
  env $J VGPU_JIT_LINK_RULE=0 VGPU_JIT_MUTANT=$m VGPU_JIT_CACHE="$w/sp53-jc.$$" nice -n 19 "$b/test_sass_chain_cont" > "$w/sp53-mut.$$" 2>&1; rc2=$?
  s=$(grep -E "BSYNCs run|FAIL|with the tier and" "$w/sp53-mut.$$" | head -2 | tr '\n' ' ' | cut -c1-260)
  rm -rf "$w/sp53-jc.$$"; mkdir -m 700 "$w/sp53-jc.$$"
  env $J VGPU_JIT_MUTANT=$m VGPU_JIT_CACHE="$w/sp53-jc.$$" nice -n 19 "$b/test_sass_chain_cont" > "$w/sp53-mut.$$" 2>&1; rc4=$?
  t=$(grep -E "left unlinked|FAIL|VCHECK" "$w/sp53-mut.$$" | head -3 | tr '\n' ' ' | cut -c1-260)
  rm -rf "$w/sp53-jc.$$" "$w/sp53-mut.$$"
  echo "mutant=${m:-none} unit rc=$rc1 rows rc=$rc3 sass rc=$rc2 rule rc=$rc4 | unit: $u | rows: $r | sass: $s | rule: $t"
}
run ""
for m in ${ONLY:-link_bsync_budget link_bsync_charge link_bsync_group link_bsync_glanes link_bsync_olo link_bsync_fl row_loop escape_outline host_bsync_olo host_bsync_pcs link_rule_never link_rule_always}; do run $m; done
[ -z "$NOLOCK" ] && flock -u 9
true
