#!/bin/bash
# mutants.sh <build with -DVGPU_JIT_MUTANTS=ON> : the fused-run test (test_jit_resid, "fused") on the pristine generator and under each mutant of
# gen_fused_run. A mutant must make the test fail (rc 1); the pristine run must pass (rc 0). Holds the machine lock; one compiler at a time.
b=$1
if [ -z "$NOLOCK" ]; then exec 9>>/tmp/pantheonsim-builder.lock; flock 9; fi
run() { VGPU_JIT_MUTANT=$1 nice -n 19 "$b/test_jit_resid" fused > /tmp/sp48-mut.$$ 2>&1; rc=$?; echo "mutant=${1:-none} rc=$rc $(grep -E 'fused:|FAIL|differs' /tmp/sp48-mut.$$ | head -3 | tr '\n' ' ')"; rm -f /tmp/sp48-mut.$$; }
run ""
for m in fuse_stale fuse_noblend fuse_nostore fuse_mov fuse_iadd3 fuse_imad fuse_lop3 fuse_lea fuse_ffma fuse_fadd fuse_fmul; do run $m; done
[ -z "$NOLOCK" ] && flock -u 9
true
