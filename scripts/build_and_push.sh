#!/bin/bash

set -e
set -x

cp ../src/video2tfrecord.py .
docker build -t ykilcher/jannet .
docker push ykilcher/jannet
rm video2tfrecord.py
