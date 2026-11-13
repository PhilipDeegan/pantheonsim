#!/bin/bash
# waits for step2.sh, then the reduced A/B with the build defaults (demote_at 4, pairs on)
H=$HOME/.cache/sp62; cd $H
while ! grep -q STEP2_DONE step2.log; do sleep 30; done
nice -n 19 python3 sp62_ab.py --base $H/base/build --new $H/new/build --corp $H/corp --real $H/new/src/tools/bench/real --sp47 $H/new/src/docs/perf-design/data/sp47 --runs 5 --long-runs 3 --lock /tmp/pantheonsim-builder.lock --json $H/ab_reduced.json --only "rodinia/(gaussian|backprop|sc_gpu|pathfinder|hotspot|lavaMD)$|polybench/(SYRK|LU|ADI|3MM|ATAX)$|hecbench/(perplexity|s3d|bh|bitonic-sort|aop|silu)-cuda$"
echo STEP3_DONE
