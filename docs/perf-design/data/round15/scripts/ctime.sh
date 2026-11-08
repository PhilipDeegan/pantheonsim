#!/bin/bash
# compile-time experiments on the kept source of the SYRK unit
cd /tmp/sp33-jit/work/kc
f=$(ls *.cpp | head -1); rt=$(ls -d rt-*)
base="-std=c++20 -fPIC -shared -fexceptions -ffp-contract=off -fvisibility=hidden -w -mavx2 -mfma -mf16c -mpopcnt -I$rt"
# a single-instantiation variant of the same source
python3 - "$f" <<'PY'
import re,sys
s=open(sys.argv[1]).read()
s=re.sub(r"  if \(group == 0xffffffffu\) body_(\d+)\(c, 0xffffffffu, 32u, entry, n0\);\n  else body_\d+\(c, group, glanes, entry, n0\);", r"  body_\1(c, group, glanes, entry, n0);", s)
open("/tmp/sp33-jit/work/single.cpp","w").write(s)
PY
t() { local label=$1; shift; /usr/bin/time -f "$label: %e s wall, %U user, %M KB" "$@" -o /tmp/sp33-jit/work/ct.so 2>&1 | tail -1; }
t "O2 unroll (as run)" g++ $base -O2 -funroll-loops $f
t "O2 unroll, single instantiation" g++ $base -O2 -funroll-loops /tmp/sp33-jit/work/single.cpp
t "O2 no unroll" g++ $base -O2 $f
t "O2 no unroll, single" g++ $base -O2 /tmp/sp33-jit/work/single.cpp
t "O1 single" g++ $base -O1 /tmp/sp33-jit/work/single.cpp
t "O1" g++ $base -O1 $f
t "O0 single" g++ $base -O0 /tmp/sp33-jit/work/single.cpp
size /tmp/sp33-jit/work/ct.so | tail -1
rm -f /tmp/sp33-jit/work/ct.so /tmp/sp33-jit/work/single.cpp
