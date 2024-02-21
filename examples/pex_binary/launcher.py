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
_SINGULARITY_CONTAINER = flags.DEFINE_string(
    "container", None, "Path to singularity container"
)


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

        resource = xm_cluster.Fileset(
            {xm.utils.resolve_path_relative_to_launcher("data.txt"): "data.txt"}
        )
        data_path = resource.get_path("data.txt", executor.Spec())  # type: ignore

        spec = xm_cluster.PexBinary(
            # This is a relative path to the launcher that contains
            # your python package (i.e. the directory that contains pyproject.toml)
            # Entrypoint is the python module that you would like to
            # In the implementation, this is translated to
            #   python3 -m py_package.main
            entrypoint=xm_cluster.ModuleName("py_package.main"),
            path=".",
            packages=["py_package"],
            dependencies=[resource],
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

        experiment.add(
            xm.Job(
                executable=executable,
                executor=executor,
                args={"seed": 1, "data": data_path},
            )
        )


if __name__ == "__main__":
    app.run(main)
