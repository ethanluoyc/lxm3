#!/bin/bash
set -e

cat > /tmp/test_mujoco.py <<EOF
import time
from absl import app

from dm_control import suite
from dm_control.suite.wrappers import pixels

def main(_):
    env = suite.load('cartpole', 'swingup')
    env = pixels.Wrapper(env)
    timestep = env.reset()
    start_time = time.time()
    steps = 0
    while not timestep.last():
        timestep = env.step(env.action_spec().generate_value())
        steps += 1
    elapsed_time = time.time() - start_time
    print("%f" % (steps / elapsed_time,))

if __name__ == '__main__':
    app.run(main)
EOF

export MUJOCO_GL=egl
echo Running with egl
EGL_FPS=$(python /tmp/test_mujoco.py)

echo Running with osmesa
export MUJOCO_GL=osmesa
OSMESA_FPS=$(python /tmp/test_mujoco.py)

SPEED_UP=$(python -c "print(int($EGL_FPS / $OSMESA_FPS))")
echo egl_fps=$EGL_FPS osmesa_fps=$OSMESA_FPS, speedup=$SPEED_UP

if [ "$SPEED_UP" -lt "2" ]; then
    echo "ERROR: EGL rendering is not working"
    exit 1
else
    echo "SUCCESS: EGL rendering is working"
fi
