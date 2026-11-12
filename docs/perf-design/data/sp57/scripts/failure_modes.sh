#!/bin/bash
# failure_modes.sh <build> <corp> <work>: what the JIT does when its surroundings are not there (round sp57). One program (rodinia/hotspot, one
# object), the JIT forced early (VGPU_JIT_THRESHOLD=100000) so that a 0.4 s run reaches the compiler; each case prints the exit status, whether
# the output equals the tier's, the wall time, the JIT's notice and its statistics line. Run under the machine lock.
b=$1; corp=$2; w=$3; D=$corp/rodinia/bin/data
P="$corp/rodinia/bin/hotspot 64 2 2000 $D/temp_64 $D/power_64 output.out"
rm -rf "$w"; mkdir -p "$w/run"; cd "$w/run" || exit 1
base() { env -i PATH="$PATH" HOME="$HOME" LD_LIBRARY_PATH=$b/shim VGPU_GPU=nvidia/rtx3060 VGPU_THREADS=1 "$@"; }
J="VGPU_JIT=1 VGPU_JIT_THRESHOLD=100000 VGPU_JIT_STATS=1"
ref() { rm -f output.out; base VGPU_QUIET=1 $P > "$w/ref.out" 2>&1; echo "tier rc=$?"; cp output.out "$w/ref.file"; }
case_() {   # case_ <name> <env...> : runs P with the given environment on top of base
  local name=$1; shift; rm -f output.out
  local t0=$(date +%s.%N)
  base "$@" $P > "$w/o.out" 2> "$w/o.err"; local rc=$?
  local t1=$(date +%s.%N)
  local same=same; cmp -s "$w/o.out" "$w/ref.out" && cmp -s output.out "$w/ref.file" || same=DIFFERENT
  printf '%-34s rc=%d output=%s wall=%.2fs\n' "$name" $rc $same $(echo "$t1 - $t0" | bc)
  grep -h -E "^pantheonsim: JIT|units \(|cache hits" "$w/o.err" | cut -c1-230 | sed 's/^/    /'
}
ref
mkdir -m 700 "$w/c1"; case_ "cold, compiler present (sync)" $J VGPU_JIT_SYNC=1 VGPU_JIT_CACHE=$w/c1
case_ "warm" $J VGPU_JIT_SYNC=1 VGPU_JIT_CACHE=$w/c1
echo "    cache: $(ls $w/c1 | wc -l) files, $(du -sk $w/c1 | cut -f1) KiB: $(ls $w/c1 | sed 's/^[0-9a-f]\{20,\}/<key>/' | sort | uniq -c | tr '\n' ' ')"
# a damaged object: the first half of the file
so=$(ls $w/c1/*.so | head -1); head -c $(( $(stat -c %s $so) / 2 )) $so > $so.t && mv $so.t $so
case_ "object cut in half" $J VGPU_JIT_SYNC=1 VGPU_JIT_CACHE=$w/c1
case_ "after it (warm again?)" $J VGPU_JIT_SYNC=1 VGPU_JIT_CACHE=$w/c1
# no compiler on PATH (the program is named by its path)
mkdir -m 700 "$w/c2"; mkdir -p "$w/emptybin"
case_ "no compiler on PATH, cold" PATH=$w/emptybin $J VGPU_JIT_SYNC=1 VGPU_JIT_CACHE=$w/c2
case_ "no compiler on PATH, warm cache" PATH=$w/emptybin $J VGPU_JIT_SYNC=1 VGPU_JIT_CACHE=$w/c1
case_ "no compiler, default threshold" PATH=$w/emptybin VGPU_JIT=1 VGPU_JIT_STATS=1 VGPU_JIT_CACHE=$w/c2
# a compiler that fails, and one that never returns (asynchronous, as by default: does the program wait for it at exit?)
printf '#!/bin/sh\ncase "$*" in *version*|*dumpmachine*) exec c++ "$@";; esac\necho boom >&2; exit 1\n' > "$w/badcxx"; chmod +x "$w/badcxx"
printf '#!/bin/sh\ncase "$*" in *version*|*dumpmachine*) exec c++ "$@";; esac\nsleep 600\n' > "$w/slowcxx"; chmod +x "$w/slowcxx"
mkdir -m 700 "$w/c3"; case_ "compiler fails" $J VGPU_JIT_SYNC=1 VGPU_JIT_CACHE=$w/c3 VGPU_JIT_CXX=$w/badcxx
mkdir -m 700 "$w/c4"; case_ "compiler hangs (async)" $J VGPU_JIT_CACHE=$w/c4 VGPU_JIT_CXX=$w/slowcxx
echo "    left behind: $(pgrep -f "$w/slowcxx" | wc -l) processes of the hanging compiler, $(ls $w/c4 | wc -l) files in its cache: $(ls $w/c4 | tr '\n' ' ')"
# home directories: read-only, missing, none; no VGPU_JIT_CACHE
mkdir -p "$w/rohome"; chmod 555 "$w/rohome"
HOME=$w/rohome case_ "HOME read-only" $J VGPU_JIT_SYNC=1
HOME=$w/nohome case_ "HOME does not exist" $J VGPU_JIT_SYNC=1
echo "    created: $(find $w/nohome -maxdepth 4 -type d 2>/dev/null | wc -l) directories under it, mode of jit/: $(stat -c %a $w/nohome/.cache/pantheonsim/jit 2>/dev/null)"
mkdir -p "$w/xdg-ww"; chmod 777 "$w/xdg-ww"; mkdir -p -m 777 "$w/ww";
case_ "cache directory world-writable" $J VGPU_JIT_SYNC=1 VGPU_JIT_CACHE=$w/ww
# four processes at once on one empty cache (asynchronous and synchronous)
for mode in "VGPU_JIT_SYNC=1" "VGPU_JIT_SYNC=0"; do
  mkdir -m 700 "$w/c5$mode"
  for i in 1 2 3 4; do (mkdir -p $w/p$i; cd $w/p$i; rm -f output.out; base VGPU_QUIET=1 $J $mode VGPU_JIT_CACHE=$w/c5$mode $P > $w/p$i.out 2> $w/p$i.err; echo $? > $w/p$i.rc; cmp -s output.out $w/ref.file || echo DIFF >> $w/p$i.rc) & done; wait
  echo "four at once ($mode): rc $(cat $w/p?.rc | tr '\n' ' ') outputs $(for i in 1 2 3 4; do cmp -s $w/p$i.out $w/ref.out && echo -n 'same ' || echo -n 'DIFFERENT '; done) cache: $(ls $w/c5$mode/*.so 2>/dev/null | wc -l) objects, $(ls $w/c5$mode | grep -c -v -E '\.(so|sum|sel|txt|h|inc)$') stray files"
  grep -h -E "cache hits" $w/p?.err | cut -c1-120 | sed 's/^/    /'
done
case_ "after four at once (warm)" $J VGPU_JIT_SYNC=1 "VGPU_JIT_CACHE=$w/c5VGPU_JIT_SYNC=1"
chmod 755 "$w/rohome"
