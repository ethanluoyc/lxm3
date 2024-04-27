from lxm3 import xm
from lxm3.xm_cluster import artifacts
from lxm3.xm_cluster.packaging import cluster


def package_for_local_executor(packageable: xm.Packageable):
    artifact_store = artifacts.get_local_artifact_store()
    return cluster._PACKAGING_ROUTER(
        packageable.executable_spec, packageable, artifact_store
    )
