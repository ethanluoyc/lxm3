import os
import tempfile
from typing import Any, Sequence

import appdirs
import fsspec
import fsspec.implementations
import fsspec.implementations.local
import rich.progress

from lxm3 import singularity
from lxm3 import xm
from lxm3._vendor.xmanager.xm import pattern_matching
from lxm3.docker import build_image
from lxm3.singularity import image_cache
from lxm3.xm_cluster import artifacts
from lxm3.xm_cluster import console
from lxm3.xm_cluster import executable_specs as cluster_executable_specs
from lxm3.xm_cluster import executables as cluster_executables
from lxm3.xm_cluster import executors
from lxm3.xm_cluster.execution import gridengine
from lxm3.xm_cluster.execution import local
from lxm3.xm_cluster.execution import slurm
from lxm3.xm_cluster.packaging import create_archive

_IMAGE_CACHE_DIR = os.path.join(appdirs.user_cache_dir("lxm3"), "image_cache")


def singularity_image_path(image_name: str):
    return os.path.join("containers", image_name)


def archive_path(archive_name: str):
    return os.path.join("archives", archive_name)


def _transfer_file_with_progress(
    artifact_store: artifacts.ArtifactStore,
    lpath: str,
    rpath: str,
) -> str:
    should_update, reason = artifact_store.should_update(lpath, rpath)

    basename = os.path.basename(lpath)
    if should_update:
        console.info(f"Transferring {basename} ({reason})")
        with (
            rich.progress.Progress(
                rich.progress.TextColumn("[progress.description]{task.description}"),
                rich.progress.BarColumn(),
                rich.progress.TaskProgressColumn(),
                rich.progress.TimeRemainingColumn(),
                rich.progress.TransferSpeedColumn(),
                console=console.console,
            ) as progress,
            progress.open(lpath, mode="rb", description=os.path.basename(lpath)) as fin,
        ):
            put_path = artifact_store.put_fileobj(fin, rpath)
        return put_path
    else:
        console.info(f"Skipped {basename}")
        return artifact_store.get_file_info(rpath).path


def _package_python_package(
    py_package: cluster_executable_specs.PythonPackage,
    packageable: xm.Packageable,
    artifact_store: artifacts.ArtifactStore,
):
    with tempfile.TemporaryDirectory() as staging:
        entrypoint_cmd, archive_name = create_archive.create_python_archive(
            staging, py_package
        )
        local_archive_path = os.path.join(staging, archive_name)
        push_archive_name = os.path.basename(local_archive_path)
        deployed_archive_path = _transfer_file_with_progress(
            artifact_store, local_archive_path, archive_path(push_archive_name)
        )

    return cluster_executables.AppBundle(
        entrypoint_command=entrypoint_cmd,
        resource_uri=deployed_archive_path,
        name=py_package.name,
        args=packageable.args,
        env_vars=packageable.env_vars,
    )


def _package_pex_binary(
    spec: cluster_executable_specs.PexBinary,
    packageable: xm.Packageable,
    artifact_store: artifacts.ArtifactStore,
):
    with tempfile.TemporaryDirectory() as staging:
        entrypoint, archive_name = create_archive.create_pex_archive(staging, spec)
        local_archive_path = os.path.join(staging, archive_name)
        push_archive_name = os.path.basename(local_archive_path)
        deployed_archive_path = _transfer_file_with_progress(
            artifact_store, local_archive_path, archive_path(push_archive_name)
        )

    return cluster_executables.AppBundle(
        entrypoint_command=entrypoint,
        resource_uri=deployed_archive_path,
        name=spec.name,
        args=packageable.args,
        env_vars=packageable.env_vars,
    )


def _package_universal_package(
    universal_package: cluster_executable_specs.UniversalPackage,
    packageable: xm.Packageable,
    artifact_store: artifacts.ArtifactStore,
):
    with tempfile.TemporaryDirectory() as staging:
        entrypoint, archive_name = create_archive.create_universal_archive(
            staging, universal_package
        )
        local_archive_path = os.path.join(staging, os.path.basename(archive_name))
        push_archive_name = os.path.basename(local_archive_path)
        deployed_archive_path = _transfer_file_with_progress(
            artifact_store, local_archive_path, archive_path(push_archive_name)
        )

    return cluster_executables.AppBundle(
        entrypoint_command=entrypoint,
        resource_uri=deployed_archive_path,
        name=universal_package.name,
        args=packageable.args,
        env_vars=packageable.env_vars,
    )


def _package_pdm_project(
    pdm_project: cluster_executable_specs.PDMProject,
    packageable: xm.Packageable,
    artifact_store: artifacts.ArtifactStore,
):
    py_package = cluster_executable_specs.PythonPackage(
        pdm_project.entrypoint,
        path=pdm_project.path,
    )
    dockerfile = build_image.pdm_dockerfile(
        pdm_project.base_image, pdm_project.lock_file
    )
    build_image.build_image_by_dockerfile_content(
        py_package.name, dockerfile, py_package.path
    )

    singularity_image = "docker-daemon://{}:latest".format(py_package.name)
    spec = cluster_executable_specs.SingularityContainer(py_package, singularity_image)
    return _package_singularity_container(spec, packageable, artifact_store)


def _package_python_container(
    python_container: cluster_executable_specs.PythonContainer,
    packageable: xm.Packageable,
    artifact_store: artifacts.ArtifactStore,
):
    py_package = cluster_executable_specs.PythonPackage(
        python_container.entrypoint, path=python_container.path
    )
    dockerfile = build_image.python_container_dockerfile(
        base_image=python_container.base_image,
        requirements=python_container.requirements,
    )
    build_image.build_image_by_dockerfile_content(
        py_package.name, dockerfile, py_package.path
    )
    singularity_image = "docker-daemon://{}:latest".format(py_package.name)
    spec = cluster_executable_specs.SingularityContainer(py_package, singularity_image)
    return _package_singularity_container(spec, packageable, artifact_store)


def _maybe_push_singularity_image(
    singularity_image: str, artifact_store: artifacts.ArtifactStore
) -> str:
    transport, _ = singularity.uri.split(singularity_image)
    # TODO(yl): Add support for other transports.
    # TODO(yl): think about keeping multiple versions of the container in the storage.
    if not transport:
        push_image_name = os.path.basename(singularity_image)
        if isinstance(
            artifact_store.filesystem, fsspec.implementations.local.LocalFileSystem
        ):
            # Do not copy SIF image if executor is local
            return os.path.realpath(singularity_image)
        else:
            return _transfer_file_with_progress(
                artifact_store,
                singularity_image,
                singularity_image_path(push_image_name),
            )
    elif transport == "docker-daemon":
        cache_image_info = image_cache.get_cached_image(
            singularity_image, cache_dir=_IMAGE_CACHE_DIR
        )
        push_image_name = singularity.uri.filename(singularity_image, "sif")
        return _transfer_file_with_progress(
            artifact_store,
            cache_image_info.blob_path,
            singularity_image_path(push_image_name),
        )
    else:
        # For other transports, just use the image as is for now.
        # TODO(yl): Consider adding support for specifying pulling behavior.
        return singularity_image


def _package_singularity_container(
    container: cluster_executable_specs.SingularityContainer,
    packageable: xm.Packageable,
    artifact_store: artifacts.ArtifactStore,
):
    executable = _PACKAGING_ROUTER(container.entrypoint, packageable, artifact_store)
    deploy_container_path = _maybe_push_singularity_image(
        container.image_path, artifact_store
    )
    executable.container_image = cluster_executables.ContainerImage(
        name=deploy_container_path,
        image_type=cluster_executables.ContainerImageType.SINGULARITY,
    )
    return executable


def _package_docker_container(
    container: cluster_executable_specs.DockerContainer,
    packageable: xm.Packageable,
    artifact_store: artifacts.ArtifactStore,
):
    executable = _PACKAGING_ROUTER(container.entrypoint, packageable, artifact_store)
    docker_image = container.image
    executable.container_image = cluster_executables.ContainerImage(
        name=docker_image,
        image_type=cluster_executables.ContainerImageType.DOCKER,
    )
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
    _package_pex_binary,
    _package_universal_package,
    _package_pdm_project,
    _package_python_container,
    _package_singularity_container,
    _package_docker_container,
    _throw_on_unknown_executable,
)


def _get_artifact_store(executor_spec: xm.ExecutorSpec) -> artifacts.ArtifactStore:
    def local_store(executor_spec: executors.LocalSpec):
        return local.client().artifact_store

    def gridengine_store(executor_spec: executors.GridEngineSpec):
        return gridengine.client().artifact_store

    def slurm_store(executor_spec: executors.SlurmSpec):
        return slurm.client().artifact_store

    return pattern_matching.match(local_store, gridengine_store, slurm_store)(
        executor_spec
    )


def packaging_router(packageable: xm.Packageable):
    artifact_store = _get_artifact_store(packageable.executor_spec)
    return _PACKAGING_ROUTER(packageable.executable_spec, packageable, artifact_store)


def package(packageables: Sequence[xm.Packageable]):
    executables = []

    for packageable in packageables:
        executables.append(packaging_router(packageable))

    return executables
