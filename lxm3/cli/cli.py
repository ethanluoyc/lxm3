#!/usr/bin/env python3
import argparse
import errno
import importlib
import os
import sys

from absl import app
from absl.flags import argparse_flags

import lxm3


def version(_):
    print(f"lxm3 {lxm3.__version__}")


def register_version_parser(parsers: argparse._SubParsersAction):
    version_parser = parsers.add_parser(
        "version",
        help="Print version.",
        inherited_absl_flags=None,  # type: ignore
    )
    version_parser.set_defaults(command=version)


def launch(args):
    launch_script = args.launch_script
    if not os.path.exists(launch_script):
        raise OSError(errno.ENOENT, f"File not found: {launch_script}")
    sys.path.insert(0, os.path.abspath(os.path.dirname(launch_script)))
    launch_module, _ = os.path.splitext(os.path.basename(launch_script))
    m = importlib.import_module(launch_module)
    argv = [launch_script, "--xm_launch_script={}".format(launch_script)] + args.args
    app.run(m.main, argv=argv)
    sys.path.pop(0)


def shell(args):
    del args
    from lxm3.cli.shell import main

    app.run(main)


def clean(args):
    from lxm3.cli.clean import run_clean

    run_clean(args.project, args.days, args.dry_run, args.force, args.type)


def register_launch_parser(parsers: argparse._SubParsersAction):
    launch_parser = parsers.add_parser(
        "launch",
        help="Launch experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        inherited_absl_flags=None,  # type: ignore
        epilog=r"""
examples:

  Launch experiment defined in "launcher.py"
  and pass extra args "--task 1" to "launcher.py".

    lxm3 launch launcher.py -- --task 1
""",
    )
    launch_parser.add_argument(
        "launch_script",
        metavar="LAUNCH_SCRIPT",
        # nargs=1,
        help="Path to launch script.",
    )
    launch_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        metavar="ARGS",
        help="Additional arguments to pass to launch script.",
    )
    launch_parser.set_defaults(command=launch)


def register_shell_parser(parsers: argparse._SubParsersAction):
    launch_parser = parsers.add_parser(
        "shell",
        help="Open a shell.",
        inherited_absl_flags=None,  # type: ignore
    )
    launch_parser.set_defaults(command=shell)


def register_clean_parser(parsers: argparse._SubParsersAction):
    clean_parser = parsers.add_parser(
        "clean",
        help="Clean job artifacts.",
        inherited_absl_flags=None,  # type: ignore
    )
    clean_parser.add_argument("--project", required=True)
    clean_parser.add_argument(
        "--dry_run",
        default=False,
        action="store_true",
    )
    clean_parser.add_argument(
        "--force",
        default=False,
        action="store_true",
    )
    clean_parser.add_argument("--days", type=float)
    clean_parser.add_argument("--type", default=None)
    clean_parser.set_defaults(command=clean)


def _parse_flags(argv):
    parser = argparse_flags.ArgumentParser(description="lxm3 experiment scheduler.")
    parser.set_defaults(command=lambda _: parser.print_help())

    subparsers = parser.add_subparsers()

    register_version_parser(subparsers)
    register_launch_parser(subparsers)
    register_shell_parser(subparsers)
    register_clean_parser(subparsers)

    args = parser.parse_args(argv[1:])
    return args


def main(args):
    args.command(args)


def entrypoint():
    app.run(main, flags_parser=_parse_flags)


if __name__ == "__main__":
    entrypoint()
