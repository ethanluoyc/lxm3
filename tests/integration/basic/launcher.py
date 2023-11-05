#!/usr/bin/env python3
import os

from absl import app
from absl import flags

from lxm3 import xm
from lxm3 import xm_cluster

_SINGULARITY_IMAGE = flags.DEFINE_string(
    "singularity_image", None, "Name of singularity image"
)
_DOCKER_IMAGE = flags.DEFINE_string("docker_image", None, "Name of docker image")


def main(_):
    with xm_cluster.create_experiment(experiment_title="basic") as experiment:
        job_requirements = xm_cluster.JobRequirements(ram=8 * xm.GB)
        executor = xm_cluster.Local(
            job_requirements,
            singularity_options=xm_cluster.SingularityOptions(),
            docker_options=xm_cluster.DockerOptions(
                extra_options=[f"--user={os.getuid()}:{os.getgid()}"]
            ),
        )

        spec = xm_cluster.PythonPackage(
            # This is a relative path to the launcher that contains
            # your python package (i.e. the directory that contains pyproject.toml)
            path=".",
            # Entrypoint is the python module that you would like to
            # In the implementation, this is translated to
            #   python3 -m py_package.main
            entrypoint=xm_cluster.ModuleName("py_package.main"),
        )

        if _SINGULARITY_IMAGE.value is not None:
            spec = xm_cluster.SingularityContainer(
                spec,
                image_path=_SINGULARITY_IMAGE.value,
            )
        elif _DOCKER_IMAGE.value is not None:
            spec = xm_cluster.DockerContainer(
                spec,
                image=_DOCKER_IMAGE.value,
            )

        [executable] = experiment.package(
            [xm.Packageable(spec, executor_spec=executor.Spec())]
        )

        experiment.add(
            xm.Job(executable=executable, executor=executor, args={"seed": 1})
        )


if __name__ == "__main__":
    app.run(main)
