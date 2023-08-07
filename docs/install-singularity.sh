#!/bin/bash
# Installing Singularity on Ubuntu.

# For ubuntu 20.04, you can install Singularity from the pre-built deb package from
# > https://github.com/sylabs/singularity/releases/

# Check that you can run singularity correctly
# singularity pull docker://godlovedc/lolcow
# singularity run lolcow_latest.sif

set -ev
ARCH='amd64'
VERSION='3.11.4'
CODENAME=$(lsb_release -s -c)
SINGULARITY_URL="https://github.com/sylabs/singularity/releases/download/v${VERSION}/singularity-ce_${VERSION}-${CODENAME}_${ARCH}.deb"

curl -O -L $SINGULARITY_URL
curl -O -L "https://github.com/sylabs/singularity/releases/download/v${VERSION}/sha256sums"

# Install (requires sudo)
sudo apt-get install $PWD/$(basename $SINGULARITY_URL)
