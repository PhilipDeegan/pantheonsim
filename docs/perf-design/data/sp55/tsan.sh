#!/bin/bash
# tsan.sh: ThreadSanitizer build (the flags of the CI job: Debug, -fsanitize=thread -g -O1) of test_exact_threads and the memory unit
# tests, and their runs under setarch -R. One build tree at a time: the Release tree is deleted first. Holds of the machine lock of at
# most 14 minutes.
H=$HOME/.cache/sp55; L=/tmp/pantheonsim-builder.lock; cd $H
T="test_exact_threads test_memory test_memfast test_memalloc test_memrmw test_memhot_sp8 test_load_batch16 test_memcopy_direct test_memrmw_lockfree test_memcount_forms test_atom_rmw"
rm -rf $H/build
flock $L nice -n 19 cmake -S src -B tsan -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=thread -g -O1" -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread" -DCUDA_ABI_INCLUDE=$HOME/.cache/pantheonsim-sp39-pgo/cuda-abi/include > tsan-cmake.log 2>&1
for i in 1 2 3 4 5 6 7 8; do
  flock $L timeout 840 nice -n 19 cmake --build tsan -j3 --target $T >> tsan-build.log 2>&1; rc=$?
  [ $rc -ne 124 ] && break
done
echo "tsan build rc=$rc"
df -h / | tail -1
[ $rc -ne 0 ] && { tail -5 tsan-build.log; exit 1; }
run() {   # env test
  out=$(flock $L env $1 TMPDIR=$H/work nice -n 19 setarch -R $H/tsan/$2 2>&1); rc=$?
  echo "$2 [$1] rc=$rc reports=$(echo "$out" | grep -c "WARNING: ThreadSanitizer") $(echo "$out" | grep "tests passed" | tail -1)"
  [ $rc -ne 0 ] && echo "$out" | grep -A12 "WARNING: ThreadSanitizer\|FAIL" | head -40
}
for f in X=1 VGPU_TIER=0 VGPU_FASTPATH=0 VGPU_EXACT_PAGES=0; do run $f test_exact_threads; done
for t in $T; do [ $t = test_exact_threads ] || run X=1 $t; done
echo TSAN_DONE
