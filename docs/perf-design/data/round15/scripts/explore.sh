#!/bin/bash
# usage: explore.sh <tag> <sides> <runs> [progs]
tag=$1; sides=$2; runs=$3
progs=${4:-polybench/SYRK,polybench/SYR2K,samples/convolutionSeparable,samples/mergeSort,samples/sortingNetworks,hecbench/fpdc-cuda,samples/interval,samples/histogram,hecbench/black-scholes-cuda,samples/jacobiCudaGraphs,polybench/LU,samples/simpleHyperQ,hecbench/shmembench-cuda,samples/reduction,samples/transpose,samples/segmentationTreeThrust,samples/cudaTensorCoreGemm,hecbench/atomicAggregate-cuda,rodinia/gaussian}
cd /tmp/sp33-jit/scripts
nohup nice -n 19 python3 sp33.py --corp /tmp/sp33-jit/corp --builds base=/tmp/sp33-jit/base/build,v1=/tmp/sp33-jit/v1/build,v2=/tmp/sp33-jit/v2/build,v3=/tmp/sp33-jit/v3/build,v4=/tmp/sp33-jit/v4/build --sides $sides --progs $progs --runs $runs --cpu 9 --out /tmp/sp33-jit/work/$tag.json --work /tmp/sp33-jit/work > /tmp/sp33-jit/work/$tag.log 2>&1 &
