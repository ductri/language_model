#!/bin/bash

docker logs -f --timestamps $(docker run -d --runtime=nvidia -e PYTHONIOENCODING=utf-8 --name="lm_$(date +"%y-%m-%d_%H_%M_%S")" \
-v `pwd`/source/:/source \
ductricse/pytorch /bin/bash -c "/source/scripts/train.sh")

