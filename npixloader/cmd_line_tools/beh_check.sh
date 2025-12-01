#!/bin/bash
# run python script

FNAME=$1
FLAG=${2:-}

if [[ "$FLAG" == '-n' ]]; then
    ./beh_check.py "$FNAME" -n
else
    ./beh_check.py "$FNAME"
fi
