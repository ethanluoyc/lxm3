#!/usr/bin/env python3
from absl import app
from absl import flags

from lxm3 import xm
from lxm3 import xm_cluster

_LAUNCH_ON_CLUSTER = flags.DEFINE_boolean(
    "launch_on_cluster", False, "Launch on cluster"
)
_SINGULARITY_CONTAINER = flags.DEFINE_string(
    "--container", "jax-cuda.sif", "Singularity container"
)


def main(_):
    with xm_cluster.create_experiment(
        experiment_title="basic", project="basic"
    ) as experiment:
        singularity_container = _SINGULARITY_CONTAINER.value
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
