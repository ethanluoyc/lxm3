import os
import tempfile
from typing import Any

from lxm3 import singularity
from lxm3 import xm
from lxm3._vendor.xmanager.xm import pattern_matching
from lxm3.xm_cluster import artifacts
from lxm3.xm_cluster import config as config_lib
from lxm3.xm_cluster import executable_specs as cluster_executable_specs
from lxm3.xm_cluster import executables as cluster_executables
from lxm3.xm_cluster.packaging import archive_builder
from lxm3.xm_cluster.packaging import container_builder


def _package_python_package(
    py_package: cluster_executable_specs.PythonPackage,
    packageable: xm.Packageable,
    artifact_store: artifacts.ArtifactStore,
):
    with tempfile.TemporaryDirectory() as staging:
        archive_name = archive_builder.create_python_archive(staging, py_package)
        local_archive_path = os.path.join(staging, archive_name)
        entrypoint_cmd = archive_builder.ENTRYPOINT_SCRIPT
        deployed_archive_path = artifact_store.deploy_resource_archive(
            local_archive_path
        )

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
    with tempfile.TemporaryDirectory() as staging:
        archive_name = archive_builder.create_universal_archive(
            staging, universal_package
        )
        local_archive_path = os.path.join(staging, os.path.basename(archive_name))
        deployed_archive_path = artifact_store.deploy_resource_archive(
            local_archive_path
        )

    return cluster_executables.Command(
        entrypoint_command=" ".join(universal_package.entrypoint),
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
    dockerfile = container_builder.pdm_dockerfile(
        pdm_project.base_image, pdm_project.lock_file
    )
    container_builder.build_docker_image(py_package.name, dockerfile, py_package.path)

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
    dockerfile = container_builder.python_container_dockerfile(
        base_image=python_container.base_image,
        requirements=python_container.requirements,
    )
    container_builder.build_docker_image(py_package.name, dockerfile, py_package.path)
    singularity_image = "docker-daemon://{}:latest".format(py_package.name)
    spec = cluster_executable_specs.SingularityContainer(py_package, singularity_image)
    return _package_singularity_container(spec, packageable, artifact_store)


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
        artifact_store.deploy_singularity_container(singularity_image)
    elif transport == "docker-daemon":
        # Try building singularity image using cache
        cache_image_path = container_builder.build_singularity_image_from_docker_daemon(
            singularity_image
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
    _package_pdm_project,
    _package_python_container,
    _package_singularity_container,
    _package_docker_container,
    _throw_on_unknown_executable,
)


def package_for_cluster_executor(packageable: xm.Packageable):
    config = config_lib.default()
    cluster_settings = config.cluster_settings()

    artifact_store = artifacts.create_artifact_store(
        cluster_settings.storage_root,
        hostname=cluster_settings.hostname,
        user=cluster_settings.user,
        project=config.project(),
        connect_kwargs=cluster_settings.ssh_config,
    )
    return _PACKAGING_ROUTER(packageable.executable_spec, packageable, artifact_store)
