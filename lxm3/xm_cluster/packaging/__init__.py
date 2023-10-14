import atexit
import datetime
import functools
import glob
import os
import shutil
import subprocess
import tempfile
from typing import Any, Sequence

from absl import logging

from lxm3 import xm
from lxm3._vendor.xmanager.xm import pattern_matching
from lxm3.xm_cluster import executable_specs as cluster_executable_specs
from lxm3.xm_cluster import executables as cluster_executables
from lxm3.xm_cluster.console import console

_ENTRYPOINT = "./entrypoint.sh"


@functools.lru_cache()
def _staging_directory():
    staging_dir = tempfile.mkdtemp(prefix="xm_cluster_staging_")
    logging.debug("Created local staging directory: %s", staging_dir)

    def remove_staging_dir():
        logging.debug("Removing local staging directory: %s", staging_dir)
        shutil.rmtree(staging_dir)

    atexit.register(remove_staging_dir)
    return staging_dir


def _create_archive(
    staging_directory: str, py_package: cluster_executable_specs.PythonPackage
):
    package_name = py_package.name
    version = datetime.datetime.now().strftime("%Y%m%d.%H%M%S")
    archive_name = f"{package_name}-{version}"
    package_dir = py_package.path
    resources = py_package.resources

    with tempfile.TemporaryDirectory() as tmpdir:
        with console.status(
            "Creating python package archive for {}".format(package_dir)
        ):
            try:
                subprocess.run(
                    [
                        "pip",
                        "install",
                        "--no-deps",
                        "--no-compile",
                        "-t",
                        tmpdir,
                        package_dir,
                        *py_package.extra_packages,
                    ],
                    text=True,
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    "Failed to package python package: {}. stdout\n{}\nstderr{}".format(
                        package_dir, e.stdout, e.stderr
                    )
                )

            # Add resources to the archive
            for resource in resources:
                for src, dst in resource.files:
                    target_file = os.path.join(tmpdir, dst)
                    if not os.path.exists(os.path.dirname(target_file)):
                        os.makedirs(os.path.dirname(target_file))
                    if not os.path.exists(target_file):
                        shutil.copy(src, target_file)
                    else:
                        raise ValueError(
                            "Additional resource overwrites existing file: %s", src
                        )

            if os.path.exists(os.path.join(tmpdir, "bin")):
                console.log('Removing "bin/" directory as these are not yet portable.')
                shutil.rmtree(os.path.join(tmpdir, "bin"))

            for f in glob.glob(os.path.join(tmpdir, "*.dist-info")):
                shutil.rmtree(f)

            entrypoint = f"""\
#!/bin/bash
export PYTHONPATH=$(dirname $0):$PYTHONPATH
{_create_entrypoint_cmds(py_package)}
"""
            with open(os.path.join(tmpdir, _ENTRYPOINT), "w") as f:
                f.write(entrypoint)
            os.chmod(f.name, 0o755)
            archive_name = shutil.make_archive(
                os.path.join(staging_directory, archive_name),
                "zip",
                tmpdir,
                verbose=True,
            )

        console.log(
            f"Created archive: [repr.path]{os.path.basename(archive_name)}[repr.path]"
        )
        return os.path.basename(archive_name)


def _create_entrypoint_cmds(python_package: cluster_executable_specs.PythonPackage):
    if isinstance(python_package.entrypoint, cluster_executable_specs.ModuleName):
        cmds = ["python3 -m {}".format(python_package.entrypoint.module_name)]
    elif isinstance(python_package.entrypoint, cluster_executable_specs.CommandList):
        cmds = python_package.entrypoint.commands
    else:
        raise ValueError("Unexpected entrypoint: {}".format(python_package.entrypoint))
    cmds = "\n".join(cmds)
    # Allow passing extra parameters to the commands.
    if not cmds.endswith(("$@", '"$@"')):
        cmds = cmds + ' "$@"'
    return cmds


def _package_python_package(
    py_package: cluster_executable_specs.PythonPackage,
    packageable: xm.Packageable,
):
    staging = tempfile.mkdtemp(dir=_staging_directory())
    archive_name = _create_archive(staging, py_package)
    local_archive_path = os.path.join(staging, archive_name)
    entrypoint_cmd = _ENTRYPOINT

    return cluster_executables.Command(
        entrypoint_command=entrypoint_cmd,
        resource_uri=local_archive_path,
        name=py_package.name,
        args=packageable.args,
        env_vars=packageable.env_vars,
    )


def _package_universal_package(
    universal_package: cluster_executable_specs.UniversalPackage,
    packageable: xm.Packageable,
):
    staging = tempfile.mkdtemp(dir=_staging_directory())
    version = datetime.datetime.now().strftime("%Y%m%d.%H%M%S")
    archive_name = f"{universal_package.name}-{version}"

    with tempfile.TemporaryDirectory(
        prefix=f"build-{archive_name}",
        dir=_staging_directory(),
    ) as build_dir:
        package_dir = universal_package.path
        build_env = {**os.environ, "BUILDDIR": build_dir}

        build_script = os.path.join(package_dir, universal_package.build_script)

        subprocess.run(
            [build_script] + universal_package.build_args,
            cwd=package_dir,
            env=build_env,
            check=True,
        )
        archive_name = shutil.make_archive(
            os.path.join(staging, archive_name), "zip", build_dir, verbose=True
        )

    return cluster_executables.Command(
        entrypoint_command=" ".join(universal_package.entrypoint),
        resource_uri=os.path.join(staging, os.path.basename(archive_name)),
        name=universal_package.name,
        args=packageable.args,
        env_vars=packageable.env_vars,
    )


def _package_singularity_container(
    container: cluster_executable_specs.SingularityContainer,
    packageable: xm.Packageable,
):
    executable = _PACKAGING_ROUTER(container.entrypoint, packageable)
    executable.singularity_image = container.image_path
    return executable


def _throw_on_unknown_executable(
    executable: Any,
    packageable: xm.Packageable,
):
    raise TypeError(
        f"Unsupported executable specification: {executable!r}. "
        f"Packageable: {packageable!r}"
    )


_PACKAGING_ROUTER = pattern_matching.match(
    _package_python_package,
    _package_universal_package,
    _package_singularity_container,
    _throw_on_unknown_executable,
)


def package(packageables: Sequence[xm.Packageable]):
    return [_PACKAGING_ROUTER(pkg.executable_spec, pkg) for pkg in packageables]
