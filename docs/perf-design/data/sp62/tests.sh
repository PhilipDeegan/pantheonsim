#!/bin/bash
# tests.sh <build dir> [tsan]: test_exact_threads in its four forms and test_exact_lines_race, each run under the machine lock
# (setarch -R for ThreadSanitizer builds); prints the result, and ThreadSanitizer's reports when there are any.
H=$HOME/.cache/sp62; L=/tmp/pantheonsim-builder.lock; B=$1; P=""; [ "$2" = tsan ] && P="setarch -R"
mkdir -p $H/work
run() {   # env test
  out=$(flock $L env $1 TMPDIR=$H/work nice -n 19 $P $B/$2 2>&1); rc=$?
  echo "$2 [$1] rc=$rc tsan_reports=$(echo "$out" | grep -c "WARNING: ThreadSanitizer") $(echo "$out" | grep -i "passed\|failed" | tail -1)"
  if [ $rc -ne 0 ] || echo "$out" | grep -q "WARNING: ThreadSanitizer"; then echo "$out" | grep -B2 -A30 "WARNING: ThreadSanitizer\|FAIL" | head -80; fi
}
for f in X=1 VGPU_TIER=0 VGPU_FASTPATH=0 VGPU_EXACT_PAGES=0; do run $f test_exact_threads; done
run X=1 test_exact_lines_race
