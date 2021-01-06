#!/bin/bash

# Bash "strict mode", to help catch problems and bugs in the shell
# script. Every bash script you write should include this. See
# http://redsymbol.net/articles/unofficial-bash-strict-mode/ for
# details.
set -euvo pipefail

# Tell apt-get we're never going to be able to give manual
# feedback:
export DEBIAN_FRONTEND=noninteractive

# Update the package listing, so we know what package exist:
apt-get update
apt-get install -y python3 python3-pip ffmpeg libgl-dev git

python3 -m pip install -U pip
pip3 install -r requirements.txt

# Delete cached files we don't need anymore:
apt-get clean
rm -rf /var/lib/apt/lists/*
