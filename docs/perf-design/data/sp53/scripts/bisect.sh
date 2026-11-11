#!/bin/bash
# bisect.sh <build> <corp> <prog, e.g. polybench/SYR2K> <work dir>: round sp53, where SYR2K's loss with linking comes from. For each variant of the linking
# features (none; all; without merging the regions into one function; without the EXIT as a region's op; without either) and each form of the row copy
# (the loop of round sp52, VGPU_JIT_MUTANT=row_loop on a build with -DVGPU_JIT_MUTANTS=ON; the four moves written out), one run fills a cache with the
# generated source kept, and the object is read: its instructions, its `rep movs`. Nothing is timed (sp33.py does that). The caller holds the machine lock.
b=$1; corp=$2; prog=$3; w=$4
here=$(dirname "$0")
for rt in loop moves; do
  for v in "unlinked:" "linked:VGPU_JIT_LINK=1" "linked-nomerge:VGPU_JIT_LINK=1,VGPU_JIT_LINK_MERGE=0" "linked-noexit:VGPU_JIT_LINK=1,VGPU_JIT_LINK_EXIT=0" \
           "linked-nomerge-noexit:VGPU_JIT_LINK=1,VGPU_JIT_LINK_MERGE=0,VGPU_JIT_LINK_EXIT=0"; do
    name=${v%%:*}; env=${v#*:}; env=${env:+$env,}VGPU_JIT_KEEP=1
    [ $rt = loop ] && env=$env,VGPU_JIT_MUTANT=row_loop
    c=$w/bis-$(basename $prog)-$name-$rt; rm -rf "$c"
    python3 "$here/one.py" --build "$b" --corp "$corp" --prog "$prog" --cache "$c" --env "$env" --runs 1 > /dev/null
    fns=0; ins=0; rep=0
    for so in "$c"/*.so; do
      d=$(objdump -d --no-show-raw-insn "$so")
      fns=$((fns + $(echo "$d" | grep -c '<vgpu_jit_r[0-9]*>:'))); ins=$((ins + $(echo "$d" | grep -c '^ ')))
      rep=$((rep + $(echo "$d" | grep -c 'rep movs')))
    done
    echo "$prog row-copy=$rt $name: functions $fns, instructions $ins, rep movs $rep, source lines $(cat "$c"/*.cpp | wc -l)"
  done
done
