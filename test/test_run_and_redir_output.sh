#!/bin/bash
testname=$1
shift
"$@" >"../../benchmark_results/results_${testname}.txt" 2>&1
exit $?
