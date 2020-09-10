#!/usr/bin/env bash

docker run --rm -it -p 6006:6006 -p 27017 --network=gptneo_omniboard --hostname=neogpt -v $PWD:/neogpt neogpt