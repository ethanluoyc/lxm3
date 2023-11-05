import datetime
import glob
import os
import shutil
import subprocess
import tempfile

from lxm3.xm_cluster import executable_specs as cluster_executable_specs
from lxm3.xm_cluster.console import console

ENTRYPOINT_SCRIPT = "./entrypoint.sh"


class PackagingError(Exception):
    """Error raised when packaging fails."""


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


def create_python_archive(
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
                        *py_package.pip_args,
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
                if e.stderr:
                    console.log("Error during packaging, stderr:", style="bold red")
                    console.log(e.stderr, style="bold red")
                raise PackagingError(
                    f"Failed to create python package from {package_dir}"
                ) from e

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
            with open(os.path.join(tmpdir, ENTRYPOINT_SCRIPT), mode="wt") as f:
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


def create_universal_archive(
    staging_directory: str, universal_package: cluster_executable_specs.UniversalPackage
):
    version = datetime.datetime.now().strftime("%Y%m%d.%H%M%S")
    archive_name = f"{universal_package.name}-{version}"

    with tempfile.TemporaryDirectory(prefix=f"build-{archive_name}") as build_dir:
        package_dir = universal_package.path
        build_env = {**os.environ, "BUILDDIR": build_dir}

        build_script = os.path.join(package_dir, universal_package.build_script)

        try:
            subprocess.run(
                [build_script] + universal_package.build_args,
                cwd=package_dir,
                env=build_env,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise PackagingError(
                "Failed to build universal package from {}".format(package_dir)
            ) from e

        archive_name = shutil.make_archive(
            os.path.join(staging_directory, archive_name),
            "zip",
            build_dir,
            verbose=True,
        )

    return os.path.basename(archive_name)
