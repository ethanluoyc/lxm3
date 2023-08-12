#!/usr/bin/env python3
from absl import app

from lxm3 import xm
from lxm3 import xm_cluster


def main(_):
    with xm_cluster.create_experiment(experiment_title="sweep") as experiment:
        requirements = xm_cluster.JobRequirements()
        executor = xm_cluster.Local(requirements=requirements)

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
        with experiment.batch():
            for i in range(2):
                experiment.add(
                    xm.Job(
                        executable=executable,
                        executor=executor,
                        # You can pass additional arguments to your executable with args
                        # This will be translated to `--seed 1`
                        # Note for booleans we currently use the absl.flags convention
                        # so {'gpu': False} will be translated to `--nogpu`
                        args={"seed": i},
                        # You can customize environment_variables as well.
                        env_vars={"TASK": str(i)},
                    )
                )

        # You can also use a job generator.
        # This is useful for example in a few cases
        # 1. if you want to configure a working directory depending on the work unit id.
        # 2. You can to dynamically compute some additional args/env_vars.
        #    one use case is to compute environment variables to be passed to wandb.
        async def make_job(work_unit: xm.WorkUnit, **args) -> None:
            work_unit.add(
                xm.Job(
                    executable=executable,
                    executor=executor,
                    args={"seed": args["i"]},
                    env_vars={"TASK": work_unit.work_unit_id},
                )
            )

        with experiment.batch():
            for i in range(2):
                experiment.add(make_job, {"i": i})


if __name__ == "__main__":
    app.run(main)
