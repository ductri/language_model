#!/bin/bash

docker stop lm_jupyter
sleep 5

./docker/jupyter.sh
