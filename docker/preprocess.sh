#!/bin/bash

docker run -d --runtime=nvidia -e PYTHONIOENCODING=utf-8  \
-v `pwd`/source/:/source \
ductricse/pytorch /bin/bash -c "/source/scripts/preprocess.sh"

