#!/bin/bash

docker run -ti --rm -e PYTHONIOENCODING=utf-8 \
-v `pwd`:/web \
-p 8002:8080 \
halverneus/static-file-server:latest
