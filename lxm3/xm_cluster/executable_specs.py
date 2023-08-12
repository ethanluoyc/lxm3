import os
import re
from typing import List, NamedTuple, Union

import attr

from lxm3.xm import job_blocks
from lxm3.xm import executables
from lxm3.xm import utils
from lxm3 import xm


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

    @property
    def name(self) -> str:
        return name_from_path(self.path)


@attr.s(auto_attribs=True)
class SingularityContainer(job_blocks.ExecutableSpec):
    entrypoint: PythonPackage
    image_path: str = attr.ib(
        converter=utils.resolve_path_relative_to_launcher, default="."
    )

    @property
    def name(self) -> str:
        return self.entrypoint.name
