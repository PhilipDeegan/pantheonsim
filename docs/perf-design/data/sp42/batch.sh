#!/bin/bash
H=$HOME/hg; M=$H/work/measure.sh; P=$H/corp/poly; R=$H/corp/rod; C=$H/corp/hec
V=1 
for n in 2DCONV 3DCONV 3MM GEMM ATAX MVT GESUMMV JACOBI2D COVAR CORR ADI GEMVER BICG JACOBI1D; do $M poly/$n $P/$n; done
for t in "backprop 16384" "gaussian -s 64" "lud -s 256 -v" "needle 256 10" "pathfinder 1000 50 20" "hotspot 64 2 20 $R/data/temp_64 $R/data/power_64 o.out" "bfs $R/data/graph4096.txt" "srad_v2 64 64 0 31 0 31 0.5 2"; do set -- $t; n=$1; shift; VERIFY=1 $M rod/$n $R/$n "$@"; done
for t in "adam-cuda" "axpby-cuda" "matrixT-cuda 512 512 1" "mandelbrot-cuda 1" "md5hash-cuda 1 1" "memcpy-cuda 64" "mask-cuda 64 64 64 1" "maxpool3d-cuda 128 128 8 1" "merge-cuda 10000 1" "lud-cuda -s 256" "matrix-rotate-cuda 256 1" "lombscargle-cuda 1"; do set -- $t; n=$1; shift; VERIFY=1 $M hec/$n $C/$n/main "$@"; done
for n in SYRK SYR2K LU 2MM FDTD-2D; do $M poly/$n $P/$n; done
$M hec/base64e-cuda $C/base64e-cuda/main
$M hec/babelstream-cuda $C/babelstream-cuda/main
SIMCAP=600 $M hec/aligned-types-cuda $C/aligned-types-cuda/main
$M poly/GRAMSCHM $P/GRAMSCHM
echo BATCH-DONE >> $H/out/results.tsv
