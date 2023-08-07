#!/usr/bin/env python3

from absl import app
from absl import flags
from ml_collections import config_flags

from lxm3 import xm
import os
from lxm3 import xm_cluster
from lxm3.xm import utils

_LAUNCH_ON_CLUSTER = flags.DEFINE_boolean(
    "launch_on_cluster", False, "Launch on cluster"
)
_CLUSTER_HOSTNAME = flags.DEFINE_string(
    "cluster_hostname", os.environ.get("LXM_CLUSTER_HOSTNAME"), ""
)
_CLUSTER_USER = flags.DEFINE_string(
    "cluster_user", os.environ.get("LXM_CLUSTER_USER"), ""
)
_CLUSTER_STAGING_DIR = flags.DEFINE_string(
    "cluster_staging_directory", os.environ.get("LXM_CLUSTER_STAGING_DIR"), ""
)
_LOCAL_STAGING_DIR = flags.DEFINE_string(
    "local_staging_directory", os.environ.get("LXM_LOCAL_STAGING_DIR"), ""
)
_NUM_TASKS = flags.DEFINE_integer("num_tasks", 1, "Number of tasks to launch")
config_flags.DEFINE_config_file("config", None)
flags.mark_flag_as_required("config")


def main(_):
    with xm_cluster.create_experiment(
        local_staging_directory=_LOCAL_STAGING_DIR.value,
        cluster_hostname=_CLUSTER_HOSTNAME.value,
        cluster_user=_CLUSTER_USER.value,
        cluster_staging_directory=_CLUSTER_STAGING_DIR.value,
    ) as experiment:
        singularity_container = utils.resolve_path_relative_to_launcher("image.sif")
        requirements = xm_cluster.JobRequirements(tmem="1G", h_vmem="1G")
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

        config_resource = xm_cluster.Fileset(
            files={config_flags.get_config_filename(flags.FLAGS["config"]): "config.py"}
        )
        packageable = xm.Packageable(
            executable_spec=xm_cluster.PythonPackage(
                path=".",
                entrypoint=xm_cluster.ModuleName("py_package.main"),
                resources=[config_resource],
            ),
            args={"config": config_resource.get_path("config.py", executor.Spec())},
            executor_spec=executor.Spec(),
        )

        [executable] = experiment.package([packageable])

        async def make_job(work_unit: xm.WorkUnit, **args):
            job = xm.Job(
                executable=executable,
                executor=executor,
                args=args,
                env_vars={"FOO": "BAR"},
            )

            work_unit.add(job)

        with experiment.batch():
            for task_id in range(_NUM_TASKS.value):
                experiment.add(make_job, {"task": task_id})


if __name__ == "__main__":
    app.run(main)
