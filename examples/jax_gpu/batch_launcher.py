#!/usr/bin/env python3
from absl import app
from absl import flags

from lxm3 import xm
from lxm3 import xm_cluster
from lxm3.contrib import ucl

_LAUNCH_ON_CLUSTER = flags.DEFINE_boolean(
    "launch_on_cluster", False, "Launch on cluster"
)
_GPU = flags.DEFINE_boolean("gpu", False, "If set, use GPU")


def main(_):
    with xm_cluster.create_experiment(experiment_title="basic") as experiment:
        if _GPU.value:
            job_requirements = xm_cluster.JobRequirements(gpu=1, ram=8 * xm.GB)
        else:
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

        packageable = xm.Packageable(
            executable_spec=xm_cluster.PythonPackage(
                # This is a relative path to the launcher that contains
                # your python package (i.e. the directory that contains pyproject.toml)
                path=".",
                # Entrypoint is the python module that you would like to
                # In the implementation, this is translated to
                #   python3 -m py_package.main
                entrypoint=xm_cluster.ModuleName("py_package.main"),
            ),
            executor_spec=executor.Spec(),
        )

        [executable] = experiment.package([packageable])

        # To submit parameter sweep by array jobs, you can use the batch context
        # Without the batch context, jobs with be submitted individually.
        parameters = [
            {"args": {"seed": i}, "env_vars": {"TASK": str(i)}} for i in range(2)
        ]
        experiment.add(
            xm_cluster.ArrayJob(
                executable=executable, executor=executor, work_list=parameters
            )
        )


if __name__ == "__main__":
    app.run(main)
