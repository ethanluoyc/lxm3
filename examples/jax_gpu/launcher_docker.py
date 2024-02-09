#!/usr/bin/env python3
"""This showcases how to create a launcher script without ever
requiring the user to manually build a singularity image.
"""
import subprocess

from absl import app
from absl import flags

from lxm3 import xm
from lxm3 import xm_cluster
from lxm3.contrib import ucl

_LAUNCH_ON_CLUSTER = flags.DEFINE_boolean(
    "launch_on_cluster", False, "Launch on cluster"
)
_GPU = flags.DEFINE_boolean("gpu", False, "If set, use GPU")


def build_image(path: str, tag: str, dockerfile: str):
    cmd = ["docker", "buildx", "build", "-t", tag, "-f", dockerfile, path]
    subprocess.run(cmd, check=True)
    return f"docker-daemon://{tag}"


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

        spec = xm_cluster.PythonPackage(
            # This is a relative path to the launcher that contains
            # your python package (i.e. the directory that contains pyproject.toml)
            path=".",
            # Entrypoint is the python module that you would like to
            # In the implementation, this is translated to
            #   python3 -m py_package.main
            entrypoint=xm_cluster.ModuleName("py_package.main"),
        )

        image_spec = build_image(
            path=".", tag="jax_gpu:latest", dockerfile="Dockerfile"
        )
        spec = xm_cluster.SingularityContainer(spec, image_path=image_spec)

        [executable] = experiment.package(
            [xm.Packageable(spec, executor_spec=executor.Spec())]
        )

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
