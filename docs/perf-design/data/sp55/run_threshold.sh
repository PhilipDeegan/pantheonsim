#!/bin/bash
# quick.sh: the base, the measured build (new) and the build with runs asked only for accesses of 16 words or more (new2), on the programs
# where per-lane accesses in pages by word made 2 and 4 exact threads slower than the base. Medians of 3, cpus 0,2,4,6.
H=$HOME/.cache/sp55; L=/tmp/pantheonsim-builder.lock
P="python3 $H/sp55_prof.py --corp $H/corp --real $H/src/tools/bench/real --cpus 0,2,4,6 --runs 3"
for p in rodinia/lavaMD rodinia/gaussian rodinia/sc_gpu polybench/ATAX polybench/GESUMMV polybench/GEMVER; do
  for x in 2 4; do
    flock $L nice -n 19 bash -c "for b in base new new2; do echo -n \"\$b \"; $P --prog $p --build $H/\$b --env VGPU_THREADS=1,VGPU_EXACT_THREADS=$x; done"
  done
done
echo done
