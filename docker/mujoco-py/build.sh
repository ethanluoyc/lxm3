#!/usr/bin/env bash

set -euo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

IMAGE_TAG=mujoco-py:latest

docker buildx build -f Dockerfile -t $IMAGE_TAG $(pwd)
docker run --runtime=nvidia --gpus=all \
    --mount type=bind,source="$HERE/tests",target=/tests,readonly \
    -it --rm $IMAGE_TAG /tests/run_tests.sh
