from typing import Sequence

from lxm3 import xm
from lxm3._vendor.xmanager.xm import pattern_matching
from lxm3.xm_cluster import executors
from lxm3.xm_cluster.packaging import cluster as cluster_packaging
from lxm3.xm_cluster.packaging import local as local_packaging


def packaging_router(
    packageable: xm.Packageable,
):
    def package_local(executable_spec: executors.LocalSpec):
        return local_packaging.package_for_local_executor(packageable)

    def package_gridengine(executable_spec: executors.GridEngineSpec):
        return cluster_packaging.package_for_cluster_executor(packageable)

    def package_slurm(executable_spec: executors.SlurmSpec):
        return cluster_packaging.package_for_cluster_executor(packageable)

    return pattern_matching.match(
        package_local,
        package_gridengine,
        package_slurm,
    )(packageable.executor_spec)


def package(packageables: Sequence[xm.Packageable]):
    executables = []

    for packageable in packageables:
        executables.append(packaging_router(packageable))

    return executables
