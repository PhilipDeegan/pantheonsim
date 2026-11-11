#!/bin/bash
# compile_cost.sh <build> <corp> <work dir> prog...: round sp53, what the compile of the linked object costs and where. For each program and each variant
# (unlinked; linked; linked without merging the regions into one function; linked without the split blocks: the mutant link_nosplit; linked without BSYNC
# as a region's op) one run on an empty cache with the generated source kept; then every kept source is compiled again with the engine's own command line
# (jit.cpp, Engine::obtain) under `time`, three times, the median user+sys seconds reported with the source's lines and the object's instructions.
# The caller holds the machine lock.
b=$1; corp=$2; w=$3; shift 3
here=$(dirname "$0")
for prog in "$@"; do
  for v in "unlinked:" "linked:VGPU_JIT_LINK=1" "linked-nomerge:VGPU_JIT_LINK=1,VGPU_JIT_LINK_MERGE=0" "linked-nosplit:VGPU_JIT_LINK=1,VGPU_JIT_MUTANT=link_nosplit" \
           "linked-nobsync:VGPU_JIT_LINK=1,VGPU_JIT_LINK_BSYNC=0" $EXTRA_VARIANTS; do
    name=${v%%:*}; env=${v#*:}; env=${env:+$env,}VGPU_JIT_KEEP=1
    c=$w/cc-$(basename $prog)-$name; rm -rf "$c"
    python3 "$here/one.py" --build "$b" --corp "$corp" --prog "$prog" --cache "$c" --env "$env" --runs 1 > /dev/null
    rt=$(ls -d "$c"/rt-* | head -1)
    tot=0; lines=0; ins=0; n=0
    for src in "$c"/*.cpp; do
      ts=""
      for i in 1 2 3; do
        t=$( { /usr/bin/time -f "%U %S" nice -n 19 g++ -std=c++20 -O2 -funroll-loops -I"$rt" -fPIC -shared -fexceptions -ffp-contract=off -fvisibility=hidden -w -fno-stack-protector \
               -fno-stack-clash-protection -fcf-protection=none -mavx2 -mfma -mf16c -mpopcnt $JIT_FLAGS -o "$c/again.so" "$src" ; } 2>&1 | tail -1 | awk '{print $1+$2}')
        ts="$ts $t"
      done
      med=$(echo $ts | tr ' ' '\n' | sort -n | sed -n 2p)
      tot=$(echo "$tot + $med" | bc -l); lines=$((lines + $(wc -l < "$src"))); ins=$((ins + $(objdump -d --no-show-raw-insn "$c/again.so" | grep -c '^ '))); n=$((n + 1))
    done
    if [ -n "$TIME_REPORT" ]; then   # the compiler's own account of the largest source, kept beside the cache
      big=$(ls -S "$c"/*.cpp | head -1)
      nice -n 19 g++ -std=c++20 -O2 -funroll-loops -I"$rt" -fPIC -shared -fexceptions -ffp-contract=off -fvisibility=hidden -w -fno-stack-protector -fno-stack-clash-protection \
        -fcf-protection=none -mavx2 -mfma -mf16c -mpopcnt $JIT_FLAGS -ftime-report -o "$c/again.so" "$big" 2> "$w/time-report-$(basename $prog)-$name.txt"
    fi
    echo "$prog $name: objects $n, source lines $lines, instructions $ins, compile s $tot"
  done
done
