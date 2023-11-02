# type: ignore

import IPython
from absl import app
from traitlets.config import Config

from lxm3.clusters import gridengine
from lxm3.xm_cluster import config as config_lib
from lxm3.xm_cluster.console import console


def main(_):
    config = config_lib.default()
    cluster_settings = config.cluster_settings()

    console.log(
        "Creating a client to cluster "
        f"{cluster_settings.user}@{cluster_settings.hostname} ..."
    )

    c = Config()
    c.InteractiveShellApp.exec_lines = [
        "%load_ext rich",
    ]

    client = gridengine.GridEngineCluster(
        cluster_settings.hostname, cluster_settings.user
    )
    IPython.start_ipython(argv=[], config=c, user_ns={"client": client})


if __name__ == "__main__":
    app.run(main)
