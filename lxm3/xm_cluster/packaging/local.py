from lxm3 import xm
from lxm3.xm_cluster import artifacts
from lxm3.xm_cluster import config as config_lib
from lxm3.xm_cluster.packaging import cluster


def package_for_local_executor(packageable: xm.Packageable):
    config = config_lib.default()
    local_settings = config_lib.default().local_settings()
    artifact_store = artifacts.LocalArtifactStore(
        local_settings.storage_root, project=config.project()
    )
    return cluster._PACKAGING_ROUTER(
        packageable.executable_spec, packageable, artifact_store
    )
