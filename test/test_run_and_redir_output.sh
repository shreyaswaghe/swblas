#!/bin/bash
testname=$1
shift
"$@" >"results_${testname}.txt" 2>&1
exit $?
