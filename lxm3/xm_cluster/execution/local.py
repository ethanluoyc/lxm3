import asyncio
import atexit
import concurrent.futures
import datetime
import os
import shutil
import subprocess
from typing import List, Optional

from absl import logging

from lxm3 import xm
from lxm3.xm_cluster import artifacts
from lxm3.xm_cluster import config as config_lib
from lxm3.xm_cluster import executables
from lxm3.xm_cluster import executors
from lxm3.xm_cluster.console import console
from lxm3.xm_cluster.execution import common
from lxm3.xm_cluster.execution import job_script

_LOCAL_EXECUTOR: Optional[concurrent.futures.ThreadPoolExecutor] = None

_TASK_OFFSET = 1
_JOB_SCRIPT_SHEBANG = "#!/usr/bin/env bash"
_TASK_ID_VAR_NAME = "SGE_TASK_ID"


def local_executor():
    global _LOCAL_EXECUTOR
    if _LOCAL_EXECUTOR is None:
        _LOCAL_EXECUTOR = concurrent.futures.ThreadPoolExecutor(1)

        def shutdown():
            if _LOCAL_EXECUTOR is not None:
                logging.debug("Shutting down local executor...")
                _LOCAL_EXECUTOR.shutdown()

        atexit.register(shutdown)
    return _LOCAL_EXECUTOR


def _is_gpu_requested(executor: executors.Local) -> bool:
    return shutil.which("nvidia-smi") is not None


def _create_job_header(
    executor: executors.Local, jobs: List[xm.Job], job_script_dir: str, job_name: str
) -> str:
    return ""


def _get_setup_cmds(
    executable: executables.Command,
    executor: executors.Local,
) -> str:
    del executor
    cmds = ["echo >&2 INFO[$(basename $0)]: Running on host $(hostname)"]

    if executable.singularity_image is not None:
        cmds.append(
            "echo >&2 INFO[$(basename $0)]: Singularity version: $(singularity --version)"
        )
    return "\n".join(cmds)


def create_job_script(
    local_settings: config_lib.LocalSettings,
    artifact: artifacts.Artifact,
    jobs: List[xm.Job],
    version: Optional[str] = None,
) -> str:
    version = version or datetime.datetime.now().strftime("%Y%m%d.%H%M%S")
    job_name = f"job-{version}"

    executable = jobs[0].executable
    executor = jobs[0].executor

    assert isinstance(executor, executors.Local)
    assert isinstance(executable, executables.Command)
    job_script.validate_same_job_configuration(jobs)

    job_script_dir = artifact.job_path(job_name)

    setup = _get_setup_cmds(executable, executor)
    header = _create_job_header(executor, jobs, job_script_dir, job_name)

    return common.create_array_job(
        executable=executable,
        singularity_image=executable.singularity_image,
        singularity_options=executor.singularity_options,
        jobs=jobs,
        use_gpu=_is_gpu_requested(executor),
        job_script_shebang=_JOB_SCRIPT_SHEBANG,
        task_offset=_TASK_OFFSET,
        task_id_var_name=_TASK_ID_VAR_NAME,
        setup=setup,
        header=header,
        settings=local_settings,
    )


class LocalExecutionHandle:
    def __init__(self, future: concurrent.futures.Future) -> None:
        self.future = future

    async def wait(self) -> None:
        return await asyncio.wrap_future(self.future)

    async def monitor(self) -> None:
        await asyncio.wrap_future(self.future)


async def launch(config: config_lib.Config, jobs: List[xm.Job]):
    if len(jobs) < 1:
        return []

    local_config = config.local_settings()
    artifact = artifacts.LocalArtifact(
        local_config.storage_root, project=config.project()
    )
    version = datetime.datetime.now().strftime("%Y%m%d.%H%M%S")
    job_script_content = create_job_script(local_config, artifact, jobs, version)
    job_name = f"job-{version}"

    job_script_dir = artifact.job_path(job_name)
    artifact.deploy_job_scripts(job_name, job_script_content)
    job_script_path = os.path.join(job_script_dir, job_script.JOB_SCRIPT_NAME)

    console.print(f"Launching {len(jobs)} jobs locally...")
    handles = []
    for i in range(len(jobs)):

        def task(i):
            subprocess.run(
                ["bash", job_script_path],
                env={**os.environ, _TASK_ID_VAR_NAME: str(i + _TASK_OFFSET)},
            )

        future = local_executor().submit(task, i)
        handles.append(LocalExecutionHandle(future))

    return handles
