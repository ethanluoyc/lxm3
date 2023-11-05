#!/usr/bin/env python3
from absl import app
from absl import flags

from lxm3 import xm
from lxm3 import xm_cluster
from lxm3.contrib import ucl

_LAUNCH_ON_CLUSTER = flags.DEFINE_boolean(
    "launch_on_cluster", False, "Launch on cluster"
)


def main(_):
    with xm_cluster.create_experiment(experiment_title="basic") as experiment:
        job_requirements = xm_cluster.JobRequirements(ram=8 * xm.GB)
        if _LAUNCH_ON_CLUSTER.value:
            # This is a special case for using SGE in UCL where we use generic
            # job requirements and translate to SGE specific requirements.
            # Non-UCL users, use `xm_cluster.GridEngine directly`.
            executor = ucl.UclGridEngine(
                job_requirements,
                walltime=10 * xm.Min,
            )
        else:
            executor = xm_cluster.Local(job_requirements)

        spec = xm_cluster.PythonContainer(
            path=".",
            base_image="python:3.10-slim",
            entrypoint=xm_cluster.ModuleName("py_package.main"),
        )

        [executable] = experiment.package(
            [xm.Packageable(spec, executor_spec=executor.Spec())]
        )

        experiment.add(xm.Job(executable=executable, executor=executor))


if __name__ == "__main__":
    app.run(main)
