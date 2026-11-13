#!/bin/bash
# tests.sh: the unit tests the change touches, at the default table and with VGPU_SIMD pinned. Run under the shared lock.
cd ~/.cache/sp54/build
for t in test_lanes512 test_simd test_tier test_sass_alu_runs test_simd_dispatch; do
  for s in default avx2 avx512; do
    if [ $s = default ]; then out=$(nice -n 19 ./$t 2>&1); else out=$(VGPU_SIMD=$s nice -n 19 ./$t 2>&1); fi
    echo "$t VGPU_SIMD=$s rc $? :: $(echo "$out" | tail -1)"
  done
done
python3 ../src/tests/lint/check_hot_unit_layout.py CMakeFiles/vgpu.dir 2>&1 | tail -2
