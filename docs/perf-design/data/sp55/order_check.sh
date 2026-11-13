#!/bin/bash
# order_check.sh: does a 2-thread run's place in a round of the A/B matter? In the A/B a round is a.t1 a.x2 a.x4 a.x8 b.t1 b.x2 ..., so the
# base's 2-thread run follows a 1-thread run and this round's follows an 8-thread run. Here the two builds' 2-thread runs alone, one after
# the other, 7 times, in both orders (the build named first runs first in each round). Medians of the wall seconds.
H=$HOME/.cache/sp55; cd $H
C="--corp $H/corp --real $H/src/tools/bench/real --sp47 $H --cpus 0,2,4,6,8,10,12,14 --runs 7 --long-runs 7 --lock /tmp/pantheonsim-builder.lock --variants a.x2,b.x2 --only aobench-cuda\$|lavaMD\$|gaussian\$"
rm -f data/order-base-first.json data/order-new-first.json
echo "base first (a = base, b = new):"
nice -n 19 python3 sp55_ab.py --base $H/base --new $H/new $C --json data/order-base-first.json | cut -c1-70
echo "new first (a = new, b = base):"
nice -n 19 python3 sp55_ab.py --base $H/new --new $H/base $C --json data/order-new-first.json | cut -c1-70
