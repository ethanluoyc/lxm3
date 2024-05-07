import datetime
import filecmp
import glob
import os
import shutil
import subprocess
import tempfile
from typing import Sequence, Tuple

from lxm3.xm_cluster import console
from lxm3.xm_cluster import executable_specs

ENTRYPOINT_SCRIPT = "./entrypoint.sh"


class PackagingError(Exception):
    """Error raised when packaging fails."""


def _create_entrypoint_cmds(python_package: executable_specs.PythonPackage) -> str:
    if isinstance(python_package.entrypoint, executable_specs.ModuleName):
        cmds = ["python3 -m {}".format(python_package.entrypoint.module_name)]
    elif isinstance(python_package.entrypoint, executable_specs.CommandList):
        cmds = python_package.entrypoint.commands
    else:
        raise ValueError("Unexpected entrypoint: {}".format(python_package.entrypoint))
    cmds = "\n".join(cmds)
    # Allow passing extra parameters to the commands.
    if not cmds.endswith(("$@", '"$@"')):
        cmds = cmds + ' "$@"'
    return cmds


def _copy_resources(
    target_dir: str, resources: Sequence[executable_specs.Fileset]
) -> None:
    # Add resources to the archive
    for resource in resources:
        for src, dst in resource.files:
            target_file = os.path.join(target_dir, dst)
            if not os.path.exists(os.path.dirname(target_file)):
                os.makedirs(os.path.dirname(target_file))
            if not os.path.exists(target_file):
                shutil.copy(src, target_file)
            else:
                # Trying to override a file in the archive,
                # check if the file contents are the same
                if not filecmp.cmp(src, target_file):
                    # Raise if contents differ
                    raise ValueError(
                        "Additional resource overwrites existing file: %s", src
                    )


def create_python_archive(
    staging_directory: str, py_package: executable_specs.PythonPackage
) -> Tuple[str, str]:
    package_name = py_package.name
    version = datetime.datetime.now().strftime("%Y%m%d.%H%M%S")
    archive_name = f"{package_name}-{version}"
    package_dir = py_package.path

    with tempfile.TemporaryDirectory() as tmpdir:
        with console.status(f"Building python package [dim]{package_dir}[/]"):
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
                    console.error("Error during packaging, stderr:")
                    console.error(e.stderr)
                raise PackagingError(
                    f"Failed to create python package from {package_dir}"
                ) from e

            # Add resources to the archive
            _copy_resources(tmpdir, py_package.resources)
            if os.path.exists(os.path.join(tmpdir, "bin")):
                console.info('Removing "bin/" directory as these are not yet portable.')
                shutil.rmtree(os.path.join(tmpdir, "bin"))

            for f in glob.glob(os.path.join(tmpdir, "*.dist-info")):
                shutil.rmtree(f)

            entrypoint = "\n".join(
                [
                    "#!/bin/bash",
                    "export PYTHONPATH=$(dirname $0):$PYTHONPATH",
                    _create_entrypoint_cmds(py_package),
                ]
            )

            with open(os.path.join(tmpdir, ENTRYPOINT_SCRIPT), mode="wt") as f:
                f.write(entrypoint)
            os.chmod(f.name, 0o755)
            archive_name = shutil.make_archive(
                os.path.join(staging_directory, archive_name),
                "zip",
                tmpdir,
                verbose=True,
            )

        console.info(f"Created archive: {os.path.basename(archive_name)}")

    return ENTRYPOINT_SCRIPT, os.path.basename(archive_name)


def create_universal_archive(
    staging_directory: str, universal_package: executable_specs.UniversalPackage
) -> Tuple[str, str]:
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

    return " ".join(universal_package.entrypoint), os.path.basename(archive_name)


def create_pex_archive(
    staging: str, spec: executable_specs.PexBinary
) -> Tuple[str, str]:
    pex_executable = shutil.which("pex")
    version = datetime.datetime.now().strftime("%Y%m%d.%H%M%S")
    archive_name = f"{spec.name}-{version}"
    pex_name = f"{spec.name}.pex"
    install_dir = os.path.join(staging, "install")
    pex_path = os.path.join(install_dir, pex_name)

    pex_options = []
    for pkg in spec.packages:
        pex_options.extend(["--package", pkg])
    for pkg in spec.modules:
        pex_options.extend(["--module", pkg])
    pex_options.extend(["--inherit-path=fallback"])
    pex_options.extend(["--entry-point", spec.entrypoint.module_name])
    pex_options.extend(["--runtime-pex-root=./.pex"])
    with console.status(f"Creating pex {pex_name}"):
        pex_cmd = [pex_executable, "-o", pex_path, *pex_options]
        console.info(f"Running pex command: {' '.join(pex_cmd)}")
        subprocess.run(pex_cmd, check=True, cwd=spec.path)

    # Add resources to the archive
    _copy_resources(install_dir, spec.dependencies)

    archive_name = shutil.make_archive(
        os.path.join(staging, archive_name), "zip", install_dir
    )

    return f"./{pex_name}", os.path.basename(archive_name)
