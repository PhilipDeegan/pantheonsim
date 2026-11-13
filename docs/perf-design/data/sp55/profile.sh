#!/bin/bash
# profile.sh <build> <tag> [env for the exact runs]: the cost of a trial on the programs of round sp55's breakdown.
# Per program: VGPU_THREADS=1, then VGPU_EXACT_THREADS=2 on two cores (wall and CPU seconds, medians of 3), then one sampled run of
# the latter (sigprof.so, 2000 Hz of process CPU time) with its samples by function. Each program holds the machine lock for itself.
H=$HOME/.cache/sp55; L=/tmp/pantheonsim-builder.lock; B=$1; T=$2; X=${3:+,$3}
P="python3 $H/sp55_prof.py --build $B --corp $H/corp --real $H/src/tools/bench/real"
for p in polybench/ATAX polybench/GESUMMV polybench/GEMVER polybench/JACOBI2D hecbench/s3d-cuda rodinia/sc_gpu polybench/JACOBI1D polybench/LU; do
  n=${p#*/}
  flock $L nice -n 19 bash -c "
    $P --prog $p --cpus 0,2 --env VGPU_THREADS=1
    $P --prog $p --cpus 0,2 --env VGPU_THREADS=1,VGPU_EXACT_THREADS=2$X --sigprof $H/sigprof.so --out $H/work/prof-$T-$n.txt ${4:+--stats $H/work/stats-$T-$n.txt}
    python3 $H/sigprof_sym.py $H/work/prof-$T-$n.txt --top 14
  "
done
echo done
