#!/bin/bash

set -e

HERE=$(realpath $(dirname $0))
LAUNCHER=$HERE/launcher.py
CONFIG_PATH=$HERE/lxm.toml

rm -rf $HERE/tmp
mkdir -p $HERE/tmp

cd $HERE/tmp

echo "Running docker tests with launcher: $LAUNCHER"
lxm3 launch $LAUNCHER --docker_image=python:3.10-slim --lxm_config $CONFIG_PATH
echo "Running singularity tests with launcher: $LAUNCHER"
lxm3 launch $LAUNCHER --singularity_image=docker://python:3.10-slim --lxm_config $CONFIG_PATH

rm -rf $HERE/tmp
