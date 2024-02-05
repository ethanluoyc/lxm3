"""Helper functions for integrating with Weights & Biases.

This module provides a helper function to configure environment variables for
logging to Weights & Biases (wandb) for jobs in an experiment.

Examples:

with xm_cluster.create_experiment("experiment") as experiment:
    log_to_wandb = wandb.configure_wandb(
        project: "my_wandb_project",
        entity: "my_wandb_entity",
        group: str = "{title}_{xid}_{wid}",
    )

    # For single jobs:
    experiment.add(log_to_wandb(xm.Job(executable, ...)))

    # For array jobs:
    experiment.add(log_to_wandb(xm_cluster.ArrayJob(executable, ...)))

"""
import copy
import functools
import logging
import subprocess
from typing import Union

from lxm3 import xm
from lxm3 import xm_cluster


@functools.lru_cache()
def _get_vcs_info():
    vcs = None
    try:
        import vcsinfo

        vcs_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
        vcs = vcsinfo.detect_vcs(vcs_root)
    except subprocess.SubprocessError:
        logging.warn("Failed to detect VCS info")
    return vcs


def configure_wandb(
    project: str,
    entity: str,
    group: str = "{title}_{xid}_{wid}",
    mode: str = "online",
):
    vcs = _get_vcs_info()
    additional_envs = {
        "WANDB_PROJECT": project,
        "WANDB_ENTITY": entity,
        "WANDB_MODE": mode,
    }

    if vcs is not None:
        if vcs.upstream_repo is not None:
            additional_envs["WANDB_GIT_REMOTE_URL"] = vcs.upstream_repo
        if vcs.id is not None:
            additional_envs["WANDB_GIT_COMMIT"] = vcs.id

    def add_wandb_env_vars(job: Union[xm.Job, xm_cluster.ArrayJob]):
        job = copy.copy(job)  # type: ignore

        async def job_gen(work_unit: xm_cluster.ClusterWorkUnit):
            experiment_title = work_unit.experiment._experiment_title
            xid = work_unit.experiment_id
            wid = work_unit.work_unit_id

            if isinstance(job, xm.Job):
                env_vars = {
                    **job.env_vars,
                    **additional_envs,
                    **{
                        "WANDB_NAME": f"{experiment_title}_{xid}_{wid}",
                        "WANDB_RUN_GROUP": group.format(
                            title=experiment_title, xid=xid, wid=wid
                        ),
                    },
                }
                return work_unit.add(
                    xm.Job(
                        executable=job.executable,
                        executor=job.executor,
                        args=job.args,
                        env_vars=env_vars,
                    )
                )

            elif isinstance(job, xm_cluster.ArrayJob):
                num_tasks = len(job.args)
                new_env_vars = [
                    {
                        **job.env_vars[task_id],
                        **additional_envs,
                        **{
                            "WANDB_NAME": f"{experiment_title}_{xid}_{wid}_{task_id + 1}",
                            "WANDB_RUN_GROUP": group.format(
                                title=experiment_title, xid=xid, wid=wid
                            ),
                        },
                    }
                    for task_id in range(num_tasks)
                ]

                return work_unit.add(
                    xm_cluster.ArrayJob(
                        executable=job.executable,
                        executor=job.executor,
                        args=job.args,
                        env_vars=new_env_vars,
                    )
                )
            else:
                raise NotImplementedError(f"Unsupported job type: {type(job)}")

        return job_gen

    return add_wandb_env_vars
