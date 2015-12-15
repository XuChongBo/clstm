#!/bin/bash
set -x
set -a

#find $1 -name '*.png' | sort -r > data.txt
#sed 1,50d data.txt > train.txt
#sed 50q data.txt > test.txt
report_every=100
save_every=10000
ntrain=1000000
dewarp=center
display_every=0
test_every=10000
hidden=100
lrate=1e-4
report_time=0
# gdb --ex run --args \
../clstmocrtrain train.txt test.txt
