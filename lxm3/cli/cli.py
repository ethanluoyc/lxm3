#!/usr/bin/env python3
import errno
import importlib
import os
import sys

from absl import app


def main(argv):
    cmd = argv[1]
    if cmd == "version":
        from lxm3 import __version__

        print(__version__)
        sys.exit(0)
    if len(argv) < 3:
        raise app.UsageError("There must be at least 2 command-line arguments")
    if cmd == "launch":
        launch_script = argv[2]
        if not os.path.exists(launch_script):
            raise OSError(errno.ENOENT, f"File not found: {launch_script}")
        sys.path.insert(0, os.path.abspath(os.path.dirname(launch_script)))
        launch_module, _ = os.path.splitext(os.path.basename(launch_script))
        m = importlib.import_module(launch_module)
        # fmt:off
        argv = [launch_script, "--xm_launch_script={}".format(launch_script)] + argv[3:]
        app.run(m.main, argv=argv)
        sys.path.pop(0)
        # fmt:on


def entrypoint():
    app.run(main)


if __name__ == "__all__":
    app.run(main)
