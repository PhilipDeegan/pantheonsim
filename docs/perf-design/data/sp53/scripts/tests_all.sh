#!/bin/bash
# tests_all.sh <build> <work dir> <log>: the JIT's unit tests and the run-loop tests in every leg ctest has for them (plain, the JIT forced on, with linking,
# VGPU_JIT_LINK_BSYNC=1 and VGPU_JIT_LINK_RULE=0, and test_sass_chain_cont with linking alone and with the loop rule on), one line a leg. The caller holds the machine lock.
b=$1; w=$2; log=$3
J="VGPU_JIT=1 VGPU_JIT_STATS=1 VGPU_JIT_STATS_FILE=/dev/null VGPU_JIT_THRESHOLD=1 VGPU_JIT_SYNC=1 VGPU_JIT_HOT_PERMILLE=0 VGPU_JIT_MIN_REGION=2 VGPU_JIT_MAX_OPS=100000000 VGPU_QUIET=1"
export TMPDIR=$w
: > "$log"
t() {
  n=$1; shift
  "$@" > "$w/t.out" 2>&1; rc=$?
  echo "$n rc $rc: $(grep -c '^\[ PASS' "$w/t.out") passed, $(grep -c '^\[ FAIL' "$w/t.out") failed | $(grep -E 'link:|rows:|linked inside|BSYNCs run|left unlinked' "$w/t.out" | sed -E 's/^ +//' | tr '\n' ';' | cut -c1-1200)" >> "$log"
}
t test_jit nice -n 19 "$b/test_jit"
t test_jit_share nice -n 19 "$b/test_jit_share"
t test_jit_resid nice -n 19 "$b/test_jit_resid"
for x in test_sass_chain_cont test_sass_store_gate test_sass_alu_runs test_sass_dispatch_counts; do
  t $x nice -n 19 "$b/$x"
  rm -rf "$w/jc"; mkdir -m 700 "$w/jc"; t ${x}_jit env $J VGPU_JIT_CACHE="$w/jc" nice -n 19 "$b/$x"
  rm -rf "$w/jc"; mkdir -m 700 "$w/jc"; t ${x}_jit_link env $J VGPU_JIT_LINK=1 VGPU_JIT_LINK_BSYNC=1 VGPU_JIT_LINK_RULE=0 VGPU_JIT_CACHE="$w/jc" nice -n 19 "$b/$x"
done
rm -rf "$w/jc"; mkdir -m 700 "$w/jc"; t test_sass_chain_cont_jit_link_nobsync env $J VGPU_JIT_LINK=1 VGPU_JIT_LINK_RULE=0 VGPU_JIT_CACHE="$w/jc" nice -n 19 "$b/test_sass_chain_cont"
rm -rf "$w/jc"; mkdir -m 700 "$w/jc"; t test_sass_chain_cont_jit_link_rule env $J VGPU_JIT_LINK=1 VGPU_JIT_LINK_BSYNC=1 VGPU_JIT_CACHE="$w/jc" nice -n 19 "$b/test_sass_chain_cont"
rm -rf "$w/jc" "$w/t.out"
