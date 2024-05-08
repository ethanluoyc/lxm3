import os
import re
from typing import List, NamedTuple, Union

import attr

from lxm3 import singularity
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
        """Create a Fileset.

        Args:
            files: A mapping from local_path -> fileset path.
        """
        files = files or {}
        self._files = {}
        for src, dst in files.items():
            self.add_path(src, dst)

    def add_path(self, src: str, dst: str):
        """Add a file to the fileset.

        Args:
            src: Path to the file on the local filesystem.
            dst: Path to the file in the package.
        """
        self._files[dst] = src

    def get_path(self, name: str, executor_spec: xm.ExecutableSpec):
        """Resolve the path to a file in fileset."""
        del executor_spec
        # Currently no-op
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
    """Python package describes an executable that can be packaged by
    ``pip install``.

    :obj:`PythonPackage` describes a python distribution that can be
    installed by ``pip install``. This is the recommended way to
    package python projects. For an introduction to python packaging,
    see
    `Packaging Python Projects <https://packaging.python.org/en/latest/tutorials/packaging-projects/>`_.

    Note:
        lxm3 uses ``pip install`` to install the package into a temporary
        directory that will be subsequently packaged into a zip archive
        that will be deployed and unzipped when jobs run on the cluster.
        However, this should be considered an implementation detail.

        lxm3 uses ``pip install --no-deps`` by default to install the package.
        This means that lxm3 **will not** install the dependencies required by
        the project. The reason is that typical ML dependencies are
        large. (e.g. TensorFlow is around 400MB)
        Therefore, you should install your dependencies
        first on the cluster (better, if you are using singularity,
        install them in the container).

        The root directory of the deployed archive will be added to
        PYTHONPATH, so you can import your package as usual. However,
        if that interferes with package imports, please open an issue
        and let us know.

    Attributes:
        entrypoint: Entrypoint for the built executable.

            Currently, only :obj:`ModuleName` and :obj:`CommandList`
            are supported.

            For :obj:`ModuleName`, this corresponds to running
            ``python3 -m <module_name>``.

            For :obj:`CommandList`, this corresponds to running a
            shell script including the commands.

        path: Path to the python project.
            This should be the path to a directory ``path`` that can
            be installed as a python package by ``pip install
            <path>``. If it's a relative path, this will be resolved
            relative to the launcher's working directory.

        resources: List of resources to package. Currently, only
            :obj:`Fileset` is supported.

        extra_packages: List of paths to additional packages that will
            be install by ``pip``.

            :attr:`extra_packages` is useful if you also
            want to include a dependency that you are also working on
            locally and you want to use your development version which
            are not installed in the runtime environment.

        pip_args: Additional options that will be passed to ``pip install``.
            Defaults to ``--no-deps --no-compile``.
    """

    entrypoint: Union[CommandList, ModuleName]
    path: str = attr.ib(converter=utils.resolve_path_relative_to_launcher, default=".")
    resources: List[Fileset] = attr.ib(converter=list, default=attr.Factory(list))
    extra_packages: List[str] = attr.ib(converter=list, default=attr.Factory(list))
    pip_args: List[str] = attr.ib(
        converter=list, default=attr.Factory(lambda: ["--no-deps", "--no-compile"])
    )

    @property
    def name(self) -> str:
        return name_from_path(self.path)


@attr.s(auto_attribs=True)
class UniversalPackage(job_blocks.ExecutableSpec):
    """Universal package describes a package that can be built by a custom build script.

    Compared to `PythonPackage`, UniversalPackage is more flexible,
    as it can be used for any language and build system not supported natived by LXM3.
    However, it requires more work to set up.

    Args:
        entrypoint: Entrypoint for the built executable.
        build_script: Path to the build script. If it's a relative path, this will be
            resolved relative to `path`.
            The build script should be an executable that can be used to produce a
            directory containing files that will be packaged into a zip archive.
            During packaging. The build script put the files into the directory
            specified by the ``BUILDDIR`` environment variable.
        build_args: Additional arguments that will be passed to the build script.
            path: Path to the project.
        path: Path to the project. If it's a relative path, this will be resolved
            relative to the launcher's working directory.

    Examples:
        See ``examples/universal_package`` for an example.

    Raises:
        ValueError: If the build script is not executable.

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
class PexBinary(job_blocks.ExecutableSpec):
    """Experimental Python executable packed as PEX. Requires PEX to be installed.

    `PexBinary` is more flexible than `PythonPackage` and does not assume that your project
    can be packaged as a python distribution.

    You should be familiar with the PEX (https://docs.pex-tool.org/) before using this.

    Notes:
        A few assumptions is currently hardcoded in LXM3:
            1. No dependencies are bundled inside the PEX. This is to avoid bundling large
            dependencies like TensorFlow. Instead, dependencies should be available
            in the runtime environment (e.g. via a Singularity container). This implies
            using `--inherit-path=fallback`.
            2. Caching extracted PEXes is disabled. This is to avoid accumulating large
            number of files in the default cache directory. Instead, we always extract
            to a temporary directory (`./.pex` in the runtime root which is a scratch directory).

    Args:
        entrypoint: Entrypoint for the built executable. Only :obj:`ModuleName` is supported.
        path: Path to the project. If it's a relative path, this will be resolved
            relative to the launcher's working directory. This path will be used
            as the working directory when building the PEX.
        packages: List of packages to include in the PEX. Packages are directories containing
            python code and will be passed to the ``pex`` command as '--package <package>'.
            Syntax for passing packages nested in a subdirectory is the same as the PEX documentation.
        modules: List of modules to include in the PEX. Modules are single python files and will
            be passed to the ``pex`` command as '--module <module>'.
            Syntax for passing modules nested in a subdirectory is the same as the PEX documentation.
        dependencies: List of dependencies to include in the PEX. These are put inside
            the _archive_ created by LXM3 as opposed to being bundled inside by the PEX
            (via -D). Useful for including configuration files, etc.

    TODOs:
        Expose more options from the pex command line interface.

    Examples:
        See ``examples/pex_binary`` for an example.

    Raises:
        ValueError: If the build script is not executable.
    """

    entrypoint: ModuleName
    path: str = attr.ib(converter=utils.resolve_path_relative_to_launcher, default=".")
    packages: List[str] = attr.ib(
        converter=list, default=attr.Factory(list), kw_only=True
    )
    modules: List[str] = attr.ib(
        converter=list, default=attr.Factory(list), kw_only=True
    )
    dependencies: List[Fileset] = attr.ib(converter=list, default=attr.Factory(list))

    @property
    def name(self) -> str:
        return name_from_path(self.path)


@attr.s(auto_attribs=True)
class SingularityContainer(job_blocks.ExecutableSpec):
    """An executable that can be executed in a Singularity container.

    Attributes:
        entrypoint: Another ExcutableSpec.
        image_path: Path to a Singularity/Apptainer image.
            The following image URLs are supported:

            1. path to a local SIF image. e.g., path/to/image.sif

               Note: If it's a relative path, this will be resolved relative
               to the launcher's working directory.

            2. image URI to a container image stored in the local
               docker daemon. e.g., docker-daemon://python:3.10

               The image will be converted and cached as a singularity SIF image.

            3. Any other URIs supported by Singularity/Apptainer.
               e.g., docker://python:3.10.
               The image URI will be passed to singularity as is.
    """

    entrypoint: Union[UniversalPackage, PythonPackage, PexBinary]
    image_path: str

    @property
    def name(self) -> str:
        return self.entrypoint.name

    def __attrs_post_init__(self):
        image_path = self.image_path
        transport, _ = singularity.uri.split(image_path)
        if transport:
            return
        if not os.path.isabs(self.image_path):
            image_path = utils.resolve_path_relative_to_launcher(image_path)
        if not os.path.exists(image_path):
            raise ValueError(
                f"Unable to find Singularity image at {image_path}"
                "If you use a relative path, it should be relative to the "
                "launcher's directory."
            )
        self.image_path = image_path


@attr.s(auto_attribs=True)
class PDMProject(job_blocks.ExecutableSpec):
    entrypoint: Union[CommandList, ModuleName]
    base_image: str
    lock_file: str = "pdm.lock"
    path: str = attr.ib(converter=utils.resolve_path_relative_to_launcher, default=".")
    resources: List[Fileset] = attr.ib(converter=list, default=attr.Factory(list))
    extra_packages: List[str] = attr.ib(converter=list, default=attr.Factory(list))
    pip_args: List[str] = attr.ib(
        converter=list, default=attr.Factory(lambda: ["--no-deps", "--no-compile"])
    )

    @property
    def name(self) -> str:
        return name_from_path(self.path)


@attr.s(auto_attribs=True)
class PythonContainer(job_blocks.ExecutableSpec):
    entrypoint: Union[CommandList, ModuleName]
    base_image: str
    requirements: str = "requirements.txt"
    path: str = attr.ib(converter=utils.resolve_path_relative_to_launcher, default=".")
    resources: List[Fileset] = attr.ib(converter=list, default=attr.Factory(list))
    extra_packages: List[str] = attr.ib(converter=list, default=attr.Factory(list))
    pip_args: List[str] = attr.ib(
        converter=list, default=attr.Factory(lambda: ["--no-deps", "--no-compile"])
    )

    @property
    def name(self) -> str:
        return name_from_path(self.path)


@attr.s(auto_attribs=True)
class DockerContainer(job_blocks.ExecutableSpec):
    """An executable that can be executed in a Singularity container.

    Attributes:
        entrypoint: Another ExcutableSpec.
        image: Name of the Docker image.
    """

    entrypoint: Union[UniversalPackage, PythonPackage, PexBinary]
    image: str

    @property
    def name(self) -> str:
        return self.entrypoint.name
