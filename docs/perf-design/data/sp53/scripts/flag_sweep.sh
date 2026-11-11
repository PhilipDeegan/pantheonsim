#!/bin/bash
# flag_sweep.sh <cache dir with a kept source (VGPU_JIT_KEEP=1)>: round sp53, the compile of one generated source with the engine's command line and one
# more flag set at a time, twice each (user+sys seconds of the faster), with the instructions of the object. Nothing is run. The caller holds the machine lock.
c=$1
src=$(ls -S "$c"/*.cpp | head -1); rt=$(ls -d "$c"/rt-* | head -1)
one() {
  best=""
  for i in 1 2; do
    t=$( { /usr/bin/time -f "%U %S" nice -n 19 g++ -std=c++20 -O2 -funroll-loops -I"$rt" -fPIC -shared -fexceptions -ffp-contract=off -fvisibility=hidden -w -fno-stack-protector \
           -fno-stack-clash-protection -fcf-protection=none -mavx2 -mfma -mf16c -mpopcnt $1 -o "$c/sweep.so" "$src" ; } 2>&1 | tail -1 | awk '{print $1+$2}')
    if [ -z "$best" ] || [ "$(echo "$t < $best" | bc -l)" = 1 ]; then best=$t; fi
  done
  echo "$best s, $(objdump -d --no-show-raw-insn "$c/sweep.so" | grep -c '^ ') instructions: ${1:-none}"
}
one ""
one "-fno-tree-pre"
one "-fno-gcse"
one "-fno-thread-jumps"
one "--param=sccvn-max-alias-queries-per-access=50"
one "--param=dse-max-alias-queries-per-store=32"
one "-fno-tree-pre --param=sccvn-max-alias-queries-per-access=50 --param=dse-max-alias-queries-per-store=32"
one "-fno-tree-dse -fno-dse"
one "-fno-tree-loop-distribute-patterns"
one "-fno-schedule-insns2"
one "-fno-tree-pre -fno-gcse -fno-thread-jumps --param=sccvn-max-alias-queries-per-access=50"
one "-fno-ipa-cp -fno-ipa-sra -fno-ipa-icf"
rm -f "$c/sweep.so"
