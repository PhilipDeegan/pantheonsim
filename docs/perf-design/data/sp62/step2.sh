#!/bin/bash
H=$HOME/.cache/sp62; cd $H
python3 hold_build.py new/build vgpucuda vgpucudart test_exact_threads test_exact_lines_race | tail -2
./tests.sh new/build
python3 hold_build.py new/tsan test_exact_threads test_exact_lines_race | tail -2
./tests.sh new/tsan tsan
echo STEP2_TESTS_DONE
nice -n 19 python3 tune.py --base $H/base/build --new $H/new/build --corp $H/corp --real $H/new/src/tools/bench/real --sp47 $H/new/src/docs/perf-design/data/sp47 --only "rodinia/gaussian|polybench/SYRK$" --threads 4,8 --runs 5 --json $H/tune.json
echo TUNE_DONE
nice -n 19 python3 mutants.py $H/new/src $H/new/build $H/mut
echo STEP2_DONE
