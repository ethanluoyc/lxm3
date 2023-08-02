#!/usr/bin/env python3

from absl import app
from absl import flags

from lxm3 import xm
from lxm3 import xm_cluster
from lxm3.xm import utils

_LAUNCH_ON_CLUSTER = flags.DEFINE_boolean(
    "launch_on_cluster", False, "Launch on cluster"
)


def main(_):
    with xm_cluster.create_experiment(
        local_staging_directory=".cache/lxm",
        cluster_hostname="beaker.cs.ucl.ac.uk",
        cluster_user="yicheluo",
        cluster_staging_directory="/home/yicheluo/lxm-test-staging",
    ) as experiment:
        singularity_container = utils.resolve_path_relative_to_launcher("jax-gpu.sif")
        requirements = xm_cluster.JobRequirements(gpu=1, tmem="8G")
        if _LAUNCH_ON_CLUSTER.value:
            executor = xm_cluster.GridEngine(
                requirements=requirements,
                walltime="0:10:0",
                singularity_container=singularity_container,
            )
        else:
            executor = xm_cluster.Local(
                requirements=requirements,
                singularity_container=singularity_container,
            )

        packageable = xm.Packageable(
            executable_spec=xm_cluster.PythonPackage(
                path=".",
                entrypoint=xm_cluster.ModuleName("py_package.main"),
            ),
            executor_spec=executor.Spec(),
        )

        [executable] = experiment.package([packageable])

        experiment.add(xm.Job(executable=executable, executor=executor))


if __name__ == "__main__":
    app.run(main)
