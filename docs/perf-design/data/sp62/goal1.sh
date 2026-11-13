#!/bin/bash
H=$HOME/.cache/sp62; cd $H
python3 hold_build.py new/tsan test_exact_threads test_exact_lines_race | tail -2
./tests.sh new/tsan tsan
python3 hold_build.py new/build test_exact_lines_race test_exact_threads | tail -2
./tests.sh new/build
nice -n 19 python3 mutants.py $H/new/src $H/new/build $H/mut controls first_reader_stores_the_record record_said_by_word_before_the_words line_read_of_written_ignored
echo GOAL1_DONE
