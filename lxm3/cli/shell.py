# type: ignore

import IPython
from absl import app
from traitlets.config import Config

from lxm3.clusters import gridengine
from lxm3.xm_cluster import config as config_lib
from lxm3.xm_cluster.console import console


def main(_):
    config = config_lib.default()
    location = config.default_cluster()
    cluster_config = config.cluster_config(location)
    hostname = cluster_config["server"]
    user = cluster_config["user"]

    console.log(f"Creating a client to cluster {user}@{hostname} ...")

    c = Config()
    c.InteractiveShellApp.exec_lines = [
        "%load_ext rich",
    ]

    client = gridengine.Client(hostname, user)
    IPython.start_ipython(argv=[], config=c, user_ns={"client": client})


if __name__ == "__main__":
    app.run(main)
