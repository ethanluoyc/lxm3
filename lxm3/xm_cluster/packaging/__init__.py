import atexit
import datetime
import functools
import glob
import os
import pathlib
import shutil
import subprocess
import tempfile
from typing import Any, Sequence

import appdirs
from absl import logging

from lxm3 import singularity
from lxm3 import xm
from lxm3._vendor.xmanager.xm import pattern_matching
from lxm3.xm_cluster import artifacts
from lxm3.xm_cluster import config as config_lib
from lxm3.xm_cluster import executable_specs as cluster_executable_specs
from lxm3.xm_cluster import executables as cluster_executables
from lxm3.xm_cluster import executors
from lxm3.xm_cluster.console import console

_ENTRYPOINT = "./entrypoint.sh"


class PackagingError(Exception):
    """Error raised when packaging fails."""


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
            with open(os.path.join(tmpdir, _ENTRYPOINT), "wt") as f:
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
    artifact: artifacts.Artifact,
):
    staging = tempfile.mkdtemp(dir=_staging_directory())
    archive_name = _create_archive(staging, py_package)
    local_archive_path = os.path.join(staging, archive_name)
    entrypoint_cmd = _ENTRYPOINT

    deployed_archive_path = artifact.deploy_resource_archive(local_archive_path)

    return cluster_executables.Command(
        entrypoint_command=entrypoint_cmd,
        resource_uri=deployed_archive_path,
        name=py_package.name,
        args=packageable.args,
        env_vars=packageable.env_vars,
    )


def _package_universal_package(
    universal_package: cluster_executable_specs.UniversalPackage,
    packageable: xm.Packageable,
    artifact: artifacts.Artifact,
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
            os.path.join(staging, archive_name), "zip", build_dir, verbose=True
        )

    local_archive_path = os.path.join(staging, os.path.basename(archive_name))
    deployed_archive_path = artifact.deploy_resource_archive(local_archive_path)

    return cluster_executables.Command(
        # TODO(yl): this is not very robust
        entrypoint_command=" ".join(universal_package.entrypoint),
        resource_uri=deployed_archive_path,
        name=universal_package.name,
        args=packageable.args,
        env_vars=packageable.env_vars,
    )


def _package_singularity_container(
    container: cluster_executable_specs.SingularityContainer,
    packageable: xm.Packageable,
    artifact: artifacts.Artifact,
):
    executable = _PACKAGING_ROUTER(container.entrypoint, packageable, artifact)

    singularity_image = container.image_path
    transport, ref = singularity.uri.split(singularity_image)
    if not transport:
        deploy_container_path = artifact.singularity_image_path(
            os.path.basename(singularity_image)
        )
        artifact.deploy_singularity_container(singularity_image)
    elif transport == "docker-daemon":
        # Try building singularity image using cache
        filename = singularity.uri.filename(singularity_image, "sif")
        build_cache_dir = pathlib.Path(appdirs.user_cache_dir("lxm3"), "singularity")
        cache_image_path = build_cache_dir / filename
        marker_file = cache_image_path.with_suffix(
            cache_image_path.suffix + ".image_id"
        )
        import docker

        client = docker.from_env()
        image_id: str = client.images.get(ref).id  # type: ignore

        should_rebuild = True
        if cache_image_path.exists():
            if marker_file.exists():
                old_image_id = pathlib.Path(marker_file).read_text().strip()
                should_rebuild = old_image_id != image_id
                if should_rebuild:
                    console.log("Image ID changed, rebuilding...")
                else:
                    console.log("Reusing cached image from", cache_image_path)
            else:
                should_rebuild = True
                console.log("Marker file does not exist, rebuilding...")
        else:
            should_rebuild = True
            console.log("Container cache does not exist, rebuilding...")

        if should_rebuild:
            cache_image_path.parent.mkdir(parents=True, exist_ok=True)
            singularity.images.build_singularity_image(
                cache_image_path, singularity_image, force=True
            )
            console.log("Cached image at", cache_image_path)
            pathlib.Path(marker_file).write_text(image_id)

        deploy_container_path = artifact.singularity_image_path(
            os.path.basename(cache_image_path)
        )
        artifact.deploy_singularity_container(str(cache_image_path))
    else:
        deploy_container_path = singularity_image

    executable.singularity_image = deploy_container_path
    return executable


def _throw_on_unknown_executable(
    executable: Any,
    packageable: xm.Packageable,
    artifact: artifacts.Artifact,
):
    del artifact
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


def _executor_packaging_router(packageable: xm.Packageable):
    def _package_for_local_executor(
        packageable: xm.Packageable, executor_spec: executors.LocalSpec
    ):
        del executor_spec

        config = config_lib.default()
        local_settings = config_lib.default().local_settings()
        artifact = artifacts.LocalArtifact(
            local_settings.storage_root, project=config.project()
        )
        return _PACKAGING_ROUTER(packageable.executable_spec, packageable, artifact)

    def _package_for_gridengine_executor(
        packageable: xm.Packageable,
        executor_spec: executors.GridEngineSpec,
    ):
        del executor_spec
        config = config_lib.default()
        cluster_settings = config.cluster_settings()

        artifact = artifacts.create_artifact_store(
            cluster_settings.storage_root,
            hostname=cluster_settings.hostname,
            user=cluster_settings.user,
            project=config.project(),
            connect_kwargs=cluster_settings.ssh_config,
        )

        return _PACKAGING_ROUTER(packageable.executable_spec, packageable, artifact)

    def _package_for_slurm_executor(
        packageable: xm.Packageable, executor_spec: executors.SlurmSpec
    ):
        del executor_spec
        config = config_lib.default()
        cluster_settings = config.cluster_settings()

        artifact = artifacts.create_artifact_store(
            cluster_settings.storage_root,
            hostname=cluster_settings.hostname,
            user=cluster_settings.user,
            project=config.project(),
            connect_kwargs=cluster_settings.ssh_config,
        )

        return _PACKAGING_ROUTER(packageable.executable_spec, packageable, artifact)

    return pattern_matching.match(
        _package_for_local_executor,
        _package_for_gridengine_executor,
        _package_for_slurm_executor,
    )(packageable, packageable.executor_spec)


def package(packageables: Sequence[xm.Packageable]):
    return [_executor_packaging_router(pkg) for pkg in packageables]
