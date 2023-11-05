import atexit
import functools
import os
import shutil
import tempfile
from typing import Any, Sequence

from absl import logging

from lxm3 import singularity
from lxm3 import xm
from lxm3._vendor.xmanager.xm import pattern_matching
from lxm3.xm_cluster import artifacts
from lxm3.xm_cluster import config as config_lib
from lxm3.xm_cluster import executable_specs as cluster_executable_specs
from lxm3.xm_cluster import executables as cluster_executables
from lxm3.xm_cluster import executors
from lxm3.xm_cluster.packaging import archive_builder
from lxm3.xm_cluster.packaging import singularity_builder


@functools.lru_cache()
def _staging_directory():
    staging_dir = tempfile.mkdtemp(prefix="xm_cluster_staging_")
    logging.debug("Created local staging directory: %s", staging_dir)

    def remove_staging_dir():
        logging.debug("Removing local staging directory: %s", staging_dir)
        shutil.rmtree(staging_dir)

    atexit.register(remove_staging_dir)
    return staging_dir


def _package_python_package(
    py_package: cluster_executable_specs.PythonPackage,
    packageable: xm.Packageable,
    artifact_store: artifacts.ArtifactStore,
):
    staging = tempfile.mkdtemp(dir=_staging_directory())
    archive_name = archive_builder.create_python_archive(staging, py_package)
    local_archive_path = os.path.join(staging, archive_name)
    entrypoint_cmd = archive_builder.ENTRYPOINT_SCRIPT

    deployed_archive_path = artifact_store.deploy_resource_archive(local_archive_path)

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
    artifact_store: artifacts.ArtifactStore,
):
    staging = tempfile.mkdtemp(dir=_staging_directory())
    archive_name = archive_builder.create_universal_archive(staging, universal_package)
    local_archive_path = os.path.join(staging, os.path.basename(archive_name))
    deployed_archive_path = artifact_store.deploy_resource_archive(local_archive_path)

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
    artifact_store: artifacts.ArtifactStore,
):
    executable = _PACKAGING_ROUTER(container.entrypoint, packageable, artifact_store)

    singularity_image = container.image_path

    transport, _ = singularity.uri.split(singularity_image)
    if not transport:
        deploy_container_path = artifact_store.singularity_image_path(
            os.path.basename(singularity_image)
        )
    elif transport == "docker-daemon":
        # Try building singularity image using cache
        cache_image_path = (
            singularity_builder.build_singularity_image_from_docker_daemon(
                singularity_image
            )
        )
        deploy_container_path = artifact_store.singularity_image_path(
            os.path.basename(cache_image_path)
        )
        artifact_store.deploy_singularity_container(str(cache_image_path))
    else:
        deploy_container_path = singularity_image

    executable.singularity_image = deploy_container_path
    return executable


def _package_docker_container(
    container: cluster_executable_specs.DockerContainer,
    packageable: xm.Packageable,
    artifact_store: artifacts.ArtifactStore,
):
    executable = _PACKAGING_ROUTER(container.entrypoint, packageable, artifact_store)
    docker_image = container.image
    executable.docker_image = docker_image
    return executable


def _throw_on_unknown_executable(
    executable: Any,
    packageable: xm.Packageable,
    artifact_store: artifacts.ArtifactStore,
):
    del artifact_store
    raise TypeError(
        f"Unsupported executable specification: {executable!r}. "
        f"Packageable: {packageable!r}"
    )


_PACKAGING_ROUTER = pattern_matching.match(
    _package_python_package,
    _package_universal_package,
    _package_singularity_container,
    _package_docker_container,
    _throw_on_unknown_executable,
)


def _create_artifact_store(executor_spec: xm.ExecutorSpec):
    def _create_local_artifact_store(spec: executors.LocalSpec):
        config = config_lib.default()
        local_settings = config_lib.default().local_settings()
        return artifacts.LocalArtifactStore(
            local_settings.storage_root, project=config.project()
        )

    def _create_cluster_artifact_store(spec):
        config = config_lib.default()
        cluster_settings = config.cluster_settings()

        return artifacts.create_artifact_store(
            cluster_settings.storage_root,
            hostname=cluster_settings.hostname,
            user=cluster_settings.user,
            project=config.project(),
            connect_kwargs=cluster_settings.ssh_config,
        )

    def _create_gridengine_artifact_store(spec: executors.GridEngineSpec):
        return _create_cluster_artifact_store(spec)

    def _create_slurm_artifact_store(spec: executors.SlurmSpec):
        return _create_slurm_artifact_store(spec)

    return pattern_matching.match(
        _create_local_artifact_store,
        _create_gridengine_artifact_store,
        _create_slurm_artifact_store,
    )(executor_spec)


def package(packageables: Sequence[xm.Packageable]):
    executables = []

    for packageable in packageables:
        artifact_store = _create_artifact_store(packageable.executor_spec)
        executables.append(
            _PACKAGING_ROUTER(packageable.executable_spec, packageable, artifact_store)
        )

    return executables
