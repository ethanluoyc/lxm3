#!/usr/bin/env bash

set -e

TEST_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo Running tests

RUN_GPU_TESTS=0
if command -v nvidia-smi &> /dev/null ;
then
    RUN_GPU_TESTS=1
fi

if [ $RUN_GPU_TESTS -eq 1 ]
then
    echo '=> Running mujoco_render_test.sh'
    . $TEST_DIR/mujoco_render_test.sh
fi

echo '=> Running mujoco_py_test.py'
python $TEST_DIR/mujoco_py_test.py
