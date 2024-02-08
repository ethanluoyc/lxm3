import asyncio
import atexit
import concurrent.futures
import functools
import os
import re
import shutil
import subprocess
from typing import List, Optional

from absl import logging

from lxm3 import xm
from lxm3.xm_cluster import array_job as array_job_lib
from lxm3.xm_cluster import artifacts
from lxm3.xm_cluster import config as config_lib
from lxm3.xm_cluster import executables
from lxm3.xm_cluster import executors
from lxm3.xm_cluster.console import console
from lxm3.xm_cluster.execution import job_script_builder

_LOCAL_EXECUTOR: Optional[concurrent.futures.ThreadPoolExecutor] = None


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


class LocalJobScriptBuilder(job_script_builder.JobScriptBuilder[executors.Local]):
    TASK_OFFSET = 1
    JOB_SCRIPT_SHEBANG = "#!/usr/bin/env bash"
    TASK_ID_VAR_NAME = "LOCAL_TASK_ID"

    @classmethod
    def _is_gpu_requested(cls, executor: executors.Local) -> bool:
        return shutil.which("nvidia-smi") is not None

    @classmethod
    def _create_setup_cmds(
        cls, executable: executables.Command, executor: executors.GridEngine
    ) -> str:
        del executor
        cmds = ["echo >&2 INFO[$(basename $0)]: Running on host $(hostname)"]

        if executable.singularity_image is not None:
            cmds.append(
                "echo >&2 INFO[$(basename $0)]: Singularity version: $(singularity --version)"
            )
        return "\n".join(cmds)

    @classmethod
    def _create_job_header(
        cls,
        executor: executors.Local,
        num_array_tasks: Optional[int],
        job_log_dir: str,
        job_name: str,
    ) -> str:
        del executor, num_array_tasks, job_log_dir, job_name
        return ""

    def build(
        self, job: job_script_builder.ClusterJob, job_name: str, job_log_dir: str
    ) -> str:
        assert isinstance(job.executor, executors.Local)
        assert isinstance(job.executable, executables.Command)
        return super().build(job, job_name, job_log_dir)


class LocalExecutionHandle:
    def __init__(self, future: concurrent.futures.Future) -> None:
        self.future = future

    async def wait(self) -> None:
        return await asyncio.wrap_future(self.future)

    async def monitor(self) -> None:
        await asyncio.wrap_future(self.future)


def _local_job_predicate(job):
    if isinstance(job, xm.Job):
        return isinstance(job.executor, executors.Local)
    elif isinstance(job, array_job_lib.ArrayJob):
        return isinstance(job.executor, executors.Local)
    else:
        raise ValueError(f"Unexpected job type: {type(job)}")


class LocalClient:
    builder_cls = LocalJobScriptBuilder

    def __init__(
        self,
        settings: Optional[config_lib.LocalSettings] = None,
        artifact_store: Optional[artifacts.LocalArtifactStore] = None,
    ) -> None:
        if settings is None:
            config = config_lib.default()
            settings = config.local_settings()
        self._settings = settings

        if artifact_store is None:
            config = config_lib.default()
            artifact_store = artifacts.LocalArtifactStore(
                self._settings.storage_root, project=config.project()
            )

        self._artifact_store = artifact_store

    def launch(self, job_name: str, job: job_script_builder.ClusterJob):
        job_name = re.sub("\\W", "_", job_name)
        job_script_dir = self._artifact_store.job_path(job_name)
        job_log_dir = self._artifact_store.job_log_path(job_name)
        builder = self.builder_cls(self._settings)
        job_script_content = builder.build(job, job_name, job_log_dir)

        self._artifact_store.deploy_job_scripts(job_name, job_script_content)
        job_script_path = os.path.join(
            job_script_dir, job_script_builder.JOB_SCRIPT_NAME
        )

        if isinstance(job, array_job_lib.ArrayJob):
            num_jobs = len(job.env_vars)
        else:
            num_jobs = 1

        console.print(f"Launching {num_jobs} jobs locally...")
        handles = []
        for i in range(num_jobs):

            def task(i):
                subprocess.run(
                    ["bash", job_script_path],
                    env={
                        **os.environ,
                        LocalJobScriptBuilder.TASK_ID_VAR_NAME: str(
                            i + LocalJobScriptBuilder.TASK_OFFSET
                        ),
                    },
                )

            future = local_executor().submit(task, i)
            handles.append(LocalExecutionHandle(future))

        return handles


@functools.lru_cache()
def client() -> LocalClient:
    return LocalClient()


async def launch(job_name: str, job: job_script_builder.ClusterJob):
    if isinstance(job, array_job_lib.ArrayJob):
        jobs = [job]  # type: ignore
    elif isinstance(job, xm.JobGroup):
        jobs: List[xm.Job] = xm.job_operators.flatten_jobs(job)
    elif isinstance(job, xm.Job):
        jobs = [job]

    jobs = [job for job in jobs if _local_job_predicate(job)]

    if not jobs:
        return []

    if len(jobs) > 1:
        raise ValueError(
            "Cannot launch a job group with multiple jobs as a single job."
        )

    return client().launch(job_name, jobs[0])
