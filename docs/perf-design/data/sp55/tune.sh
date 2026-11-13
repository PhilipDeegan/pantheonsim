#!/bin/bash
# tune.sh: what each mechanism of round sp55 is worth, on two cores (cpus 0,2), CPU and wall seconds (medians of 3; LU one run).
# The build under test carries VGPU_EXACT_TUNE (1: the worker remembers 64 pages by their low bits, 2: no memory of runs of words,
# 4: no pairs of words) and VGPU_EXACT_PAGES (0: every page by word, 2: no refinement by written pages).
H=$HOME/.cache/sp55; L=/tmp/pantheonsim-builder.lock
X=VGPU_THREADS=1,VGPU_EXACT_THREADS=2
for p in polybench/ATAX polybench/GESUMMV polybench/GEMVER polybench/JACOBI2D hecbench/s3d-cuda rodinia/sc_gpu polybench/JACOBI1D polybench/LU; do
  r=3; [ $p = polybench/LU ] && r=1
  P="python3 $H/sp55_prof.py --corp $H/corp --real $H/src/tools/bench/real --prog $p --cpus 0,2 --runs $r"
  flock $L nice -n 19 bash -c "
    $P --build $H/base --env VGPU_THREADS=1
    $P --build $H/base --env VGPU_THREADS=2
    $P --build $H/base --env $X
    $P --build $H/build --env $X,VGPU_EXACT_PAGES=0,VGPU_EXACT_TUNE=6
    $P --build $H/build --env $X,VGPU_EXACT_PAGES=0
  "
  flock $L nice -n 19 bash -c "
    $P --build $H/build --env $X,VGPU_EXACT_TUNE=7,VGPU_EXACT_PAGES=2
    $P --build $H/build --env $X,VGPU_EXACT_TUNE=1
    $P --build $H/build --env $X,VGPU_EXACT_TUNE=2
    $P --build $H/build --env $X,VGPU_EXACT_TUNE=4
    $P --build $H/build --env $X,VGPU_EXACT_PAGES=2
    $P --build $H/build --env $X --stats $H/work/stats-tune-${p#*/}.txt
  "
done
echo done
