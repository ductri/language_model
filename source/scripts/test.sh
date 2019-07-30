#!/bin/bash

export PYTHONPATH=`pwd`/main
find ./main/ -type f -name "*__test.py" -exec echo "Test file {}" \; -exec python {} \;
