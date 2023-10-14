import os
import re
from typing import List, NamedTuple, Union

import attr

from lxm3 import xm
from lxm3.xm import executables
from lxm3.xm import job_blocks
from lxm3.xm import utils


def name_from_path(path: str) -> str:
    """Returns a safe to use executable name based on a filesystem path."""
    return re.sub("\\W", "_", os.path.basename(path.rstrip(os.sep)))


class Fileset(executables.BinaryDependency):
    """Additional files to package."""

    def __init__(self, files=None):
        # files is a mapping from local_path -> fileset path
        files = files or {}
        self._files = {}
        for src, dst in files.items():
            self.add_path(src, dst)

    def add_path(self, src: str, dst: str):
        self._files[dst] = src

    def get_path(self, name: str, executor_spec: xm.ExecutableSpec):
        del executor_spec
        return name

    @property
    def files(self):
        return [(src, dst) for dst, src in self._files.items()]


class ModuleName(NamedTuple):
    """Name of python module to execute when entering this project."""

    module_name: str


class CommandList(NamedTuple):
    """List of commands to execute when entering this project."""

    commands: List[str]


@attr.s(auto_attribs=True)
class PythonPackage(job_blocks.ExecutableSpec):
    entrypoint: Union[CommandList, ModuleName]
    path: str = attr.ib(converter=utils.resolve_path_relative_to_launcher, default=".")
    resources: List[Fileset] = attr.ib(converter=list, default=attr.Factory(list))
    extra_packages: List[str] = attr.ib(converter=list, default=attr.Factory(list))

    @property
    def name(self) -> str:
        return name_from_path(self.path)


@attr.s(auto_attribs=True)
class UniversalPackage(job_blocks.ExecutableSpec):
    """Universal package describes a package that can be built by a custom build script.

    Compared to `PythonPackage`, UniversalPackage is more flexible,
    as it can be used for any language and build system not supported natived by LXM3.
    However, it requires more work to set up.

    Attributes:
        entrypoint: Entrypoint for the built executable.
        build_script: Path to the build script. If it's a relative path, this will be
            resolved relative to `path`.
            The build script should be an executable that can be used to produce a
            directory containing files that will be packaged into a zip archive.
            During packaging. The build script put the files into the directory
            specified by the `BUILDDIR` environment variable.
        build_args: Additional arguments that will be passed to the build script.
            path: Path to the project.
        path: Path to the project. If it's a relative path, this will be resolved
            relative to the launcher's working directory.

    Examples:
        See `examples/universal_package` for an example.

    """

    entrypoint: List[str]
    build_script: str
    build_args: List[str] = attr.ib(converter=list, default=attr.Factory(list))
    path: str = attr.ib(converter=utils.resolve_path_relative_to_launcher, default=".")

    @property
    def name(self) -> str:
        return name_from_path(self.path)

    def __attrs_post_init__(self):
        self.build_script = os.path.join(self.path, self.build_script)
        if not os.access(self.build_script, os.X_OK):
            raise ValueError(
                "Build script is not executable"
                f"You may need to run `chmod +x {self.build_script}` to make it executable."
            )


@attr.s(auto_attribs=True)
class SingularityContainer(job_blocks.ExecutableSpec):
    entrypoint: Union[UniversalPackage, PythonPackage]
    image_path: str = attr.ib(
        converter=utils.resolve_path_relative_to_launcher, default="."
    )

    @property
    def name(self) -> str:
        return self.entrypoint.name
