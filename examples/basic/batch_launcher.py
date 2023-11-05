#!/usr/bin/env python3
from absl import app

from lxm3 import xm
from lxm3 import xm_cluster


def main(_):
    with xm_cluster.create_experiment(experiment_title="basic") as experiment:
        executor = xm_cluster.Local()

        spec = xm_cluster.PythonPackage(
            # This is a relative path to the launcher that contains
            # your python package (i.e. the directory that contains pyproject.toml)
            path=".",
            # Entrypoint is the python module that you would like to
            # In the implementation, this is translated to
            #   python3 -m py_package.main
            entrypoint=xm_cluster.ModuleName("py_package.main"),
        )

        [executable] = experiment.package(
            [xm.Packageable(spec, executor_spec=executor.Spec())]
        )

        args = [{"seed": seed} for seed in range(2)]
        env_vars = [{"TASK": f"foo_{seed}"} for seed in range(2)]
        experiment.add(
            xm_cluster.ArrayJob(
                executable=executable, executor=executor, args=args, env_vars=env_vars
            )
        )


if __name__ == "__main__":
    app.run(main)
