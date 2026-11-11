#!/bin/bash
H=$HOME/hg; M=$H/work/measure.sh; C=$H/corp/hec; S=$H/work/sens
until grep -q BATCH-DONE $H/out/results.tsv; do sleep 5; done
VERIFY=1 $M hec/adam-cuda $C/adam-cuda/main 10000 200 100
VERIFY=1 $M hec/axpby-cuda $C/axpby-cuda/main 1000
for t in "compute 1000 4096" "compute 100 65536" "compute 10 1048576" "compute 4 4194304" "compute 1 1048576" "compute 1 4096" \
         "mem 2000 4096" "mem 200 262144" "mem 50 4194304" "mem 20 16777216" "mem 1 16777216" "mem 4 4194304"; do
  set -- $t; $M sens/$1-L$2-n$3 $S "$@"
done
echo BATCH2-DONE >> $H/out/results.tsv
