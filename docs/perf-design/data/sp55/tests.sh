#!/bin/bash
# tests.sh <build>: test_exact_threads in its four forms.
H=$HOME/.cache/sp55; B=${1:-$H/build}
for f in X=1 VGPU_TIER=0 VGPU_FASTPATH=0 VGPU_EXACT_PAGES=0; do
  out=$(env $f TMPDIR=$H/work nice -n 19 $B/test_exact_threads 2>&1); rc=$?
  echo "test_exact_threads [$f] rc=$rc $(echo "$out" | tail -1)"
  [ $rc -ne 0 ] && echo "$out" | grep -v "^\[ *OK\|^ok\|PASS" | head -20
done
