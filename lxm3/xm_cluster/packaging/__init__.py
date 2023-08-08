import atexit
import datetime
import functools
import glob
import os
import shutil
import subprocess
import tempfile
import zipfile
from typing import Any, List

from absl import logging

from lxm3 import xm
from lxm3._vendor.xmanager.xm import pattern_matching
from lxm3.xm_cluster import executable_specs as cluster_executable_specs
from lxm3.xm_cluster import executables as cluster_executables
from lxm3.xm_cluster.console import console


@functools.lru_cache()
def _staging_directory():
    staging_dir = tempfile.mkdtemp(prefix="xm_cluster_staging_")
    logging.debug("Created local staging directory: %s", staging_dir)

    def remove_staging_dir():
        logging.debug("Removing local staging directory: %s", staging_dir)
        shutil.rmtree(staging_dir)

    atexit.register(remove_staging_dir)
    return staging_dir


def _create_archive(staging_directory, package_name, version, package_dir, resources):
    archive_name = f"{package_name}-{version}"

    with tempfile.TemporaryDirectory() as tmpdir:
        with console.status(
            "Creating python package archive for {}".format(package_dir)
        ):
            try:
                process = subprocess.run(
                    [
                        "pip",
                        "install",
                        "--no-deps",
                        "--no-compile",
                        "-t",
                        tmpdir,
                        package_dir,
                    ],
                    text=True,
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError:
                raise RuntimeError(
                    "Failed to package python package: {}. stdout\n{}\nstderr{}".format(
                        package_dir, process.stdout, process.stderr
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

            archive_name = shutil.make_archive(
                os.path.join(staging_directory, archive_name),
                "zip",
                tmpdir,
                verbose=True,
            )
        console.log(f"Created archive: [repr.path]{os.path.basename(archive_name)}[repr.path]")
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
    executable_spec = py_package
    package_name = executable_spec.name
    version = datetime.datetime.now().strftime("%Y%m%d.%H%M%S")

    executable_spec: cluster_executable_specs.PythonPackage = executable_spec

    staging = tempfile.mkdtemp(dir=_staging_directory())
    archive_name = _create_archive(
        staging,
        package_name,
        version,
        executable_spec.path,
        executable_spec.resources,
    )
    local_archive_path = os.path.join(staging, archive_name)
    with zipfile.ZipFile(local_archive_path, mode="a") as zf:
        if zipfile.Path(zf, "entrypoint.sh").exists():
            raise ValueError("Unexpected entrypoint.sh in the archive")
        info = zipfile.ZipInfo("entrypoint.sh")
        info.external_attr = 0o755 << 16
        entrypoint = f"""\
#!/bin/bash
{_create_entrypoint_cmds(executable_spec)}
"""
        zf.writestr(info, entrypoint)

    entrypoint_cmd = "./entrypoint.sh"

    return cluster_executables.Command(
        entrypoint_command=entrypoint_cmd,
        resource_uri=local_archive_path,
        name=package_name,
        args=packageable.args,
        env_vars=packageable.env_vars,
    )


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
    _throw_on_unknown_executable,
)


def package(packageables: List[xm.Packageable]):
    return [_PACKAGING_ROUTER(pkg.executable_spec, pkg) for pkg in packageables]
