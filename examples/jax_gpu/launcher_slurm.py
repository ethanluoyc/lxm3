#!/usr/bin/env python3
from absl import app
from absl import flags

from lxm3 import xm
from lxm3 import xm_cluster

_SINGULARITY_CONTAINER = flags.DEFINE_string(
    "container", "jax-cuda.sif", "Path to singularity container"
)
_BATCH = flags.DEFINE_bool("batch", False, "If set, launch array job example")


def main(_):
    with xm_cluster.create_experiment(experiment_title="basic") as experiment:
        executor: xm_cluster.Slurm = xm_cluster.Slurm(walltime=10 * xm.Min)

        spec = xm_cluster.PythonPackage(
            # This is a relative path to the launcher that contains
            # your python package (i.e. the directory that contains pyproject.toml)
            path=".",
            # Entrypoint is the python module that you would like to
            # In the implementation, this is translated to
            #   python3 -m py_package.main
            entrypoint=xm_cluster.ModuleName("py_package.main"),
        )

        # Wrap the python_package to be executing in a singularity container.
        singularity_container = _SINGULARITY_CONTAINER.value

        # It's actually not necessary to use a container, without it, we
        # fallback to the current python environment for local executor and
        # whatever Python environment picked up by the cluster for GridEngine.
        # For remote execution, using the host environment is not recommended.
        # as you may spend quite some time figuring out dependency problems than
        # writing a simple Dockfiler/Singularity file.
        if singularity_container is not None:
            spec = xm_cluster.SingularityContainer(
                spec,
                image_path=singularity_container,
            )

        [executable] = experiment.package(
            [xm.Packageable(spec, executor_spec=executor.Spec())]
        )

        if not _BATCH.value:
            experiment.add(
                xm.Job(
                    executable=executable,
                    executor=executor,
                    # You can pass additional arguments to your executable with args
                    # This will be translated to `--seed 1`
                    # Note for booleans we currently use the absl.flags convention
                    # so {'gpu': False} will be translated to `--nogpu`
                    args={"seed": 1},
                    # You can customize environment_variables as well.
                    env_vars={"XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
                )
            )
        else:
            with experiment.batch():
                for _ in range(2):
                    experiment.add(
                        xm.Job(
                            executable=executable,
                            executor=executor,
                            # You can pass additional arguments to your executable with args
                            # This will be translated to `--seed 1`
                            # Note for booleans we currently use the absl.flags convention
                            # so {'gpu': False} will be translated to `--nogpu`
                            args={"seed": 1},
                            # You can customize environment_variables as well.
                            env_vars={"XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
                        )
                    )


if __name__ == "__main__":
    app.run(main)
