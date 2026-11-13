#!/bin/bash
cd ~/.cache/sp46
B=~/.cache/sp46/build; C=~/.cache/sp46-corp
mkdir -p prof
run() { name=$1; dir=$2; shift 2
  (cd $dir && env VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 VGPU_THREADS=1 LD_LIBRARY_PATH=$B/shim taskset -c 6 env LD_PRELOAD=$HOME/.cache/sp46/prof.so PROF_OUT=$HOME/.cache/sp46/prof/$name.raw "$@" > /dev/null 2>&1)
  python3 sym.py prof/$name.raw 22 > prof/$name.txt; }
S=$C/samples/Samples
run radixSortThrust $C/samples/$(dirname $(awk '/radixSortThrust/{print $2}' $C/samples/sim_main.txt)) $C/samples/$(awk '/radixSortThrust/{print $2}' $C/samples/sim_main.txt)
for p in alignedTypes simpleHyperQ segmentationTreeThrust; do e=$(awk -v p=$p '$2 ~ "/"p"$"{print $2}' $C/samples/sim_main.txt); run $p $C/samples/$(dirname $e) $C/samples/$e; done
run LU $C/polybench/bin $C/polybench/bin/LU
