import asyncio
import atexit
import concurrent.futures
import functools
import os
import re
import shutil
import subprocess
from typing import Optional

import fsspec
from absl import logging

from lxm3 import xm
from lxm3.xm_cluster import array_job as array_job_lib
from lxm3.xm_cluster import artifacts
from lxm3.xm_cluster import config as config_lib
from lxm3.xm_cluster import console
from lxm3.xm_cluster import executables
from lxm3.xm_cluster import executors
from lxm3.xm_cluster.execution import job_script_builder

_LOCAL_EXECUTOR: Optional[concurrent.futures.ThreadPoolExecutor] = None


def local_executor():
    global _LOCAL_EXECUTOR
    if _LOCAL_EXECUTOR is None:
        _LOCAL_EXECUTOR = concurrent.futures.ThreadPoolExecutor(1)

        def shutdown(*args, **kwargs):
            if _LOCAL_EXECUTOR is not None:
                logging.debug("Shutting down local executor...")
                _LOCAL_EXECUTOR.shutdown()

        atexit.register(shutdown)
    return _LOCAL_EXECUTOR


class LocalJobScriptBuilder(job_script_builder.JobScriptBuilder[executors.Local]):
    ARRAY_TASK_ID = "LOCAL_TASK_ID"
    ARRAY_TASK_OFFSET = 1
    JOB_SCRIPT_SHEBANG = "#!/usr/bin/env bash"
    JOB_ENV_PATTERN = "^(LOCAL_TASK_ID)"

    @classmethod
    def _is_gpu_requested(cls, executor: executors.Local) -> bool:
        return shutil.which("nvidia-smi") is not None

    @classmethod
    def _create_job_script_prologue(
        cls, executable: executables.AppBundle, executor: executors.Local
    ) -> str:
        del executor
        cmds = ['echo >&2 "INFO[$(basename "$0")]: Running on host $(hostname)"']
        return "\n".join(cmds)

    @classmethod
    def _create_job_script_header(
        cls,
        executor: executors.Local,
        num_array_tasks: Optional[int],
        job_log_dir: str,
        job_name: str,
    ) -> str:
        del executor, num_array_tasks, job_log_dir, job_name
        return ""

    def build(
        self, job: job_script_builder.JobType, job_name: str, job_log_dir: str
    ) -> str:
        assert isinstance(job.executor, executors.Local)
        assert isinstance(job.executable, executables.AppBundle)
        return super().build(job, job_name, job_log_dir)


class LocalExecutionHandle:
    def __init__(self, future: concurrent.futures.Future) -> None:
        self.future = future

    async def wait(self) -> None:
        return await asyncio.wrap_future(self.future)

    async def monitor(self) -> None:
        await asyncio.wrap_future(self.future)


class LocalClient:
    builder_cls: type[LocalJobScriptBuilder] = LocalJobScriptBuilder

    def __init__(
        self,
        settings: config_lib.LocalSettings,
        artifact_store: artifacts.ArtifactStore,
    ) -> None:
        self._settings = settings
        self._artifact_store = artifact_store

    @property
    def artifact_store(self):
        return self._artifact_store

    def launch(self, job_name: str, job: job_script_builder.JobType):
        job_name = re.sub("\\W", "_", job_name)

        job_log_dir = job_script_builder.job_log_path(job_name)
        self._artifact_store.ensure_dir(job_log_dir)
        job_log_dir = self._artifact_store.normalize_path(job_log_dir)
        builder = self.builder_cls()
        job_script_content = builder.build(job, job_name, job_log_dir)
        job_script_path = self._artifact_store.put_text(
            job_script_content, job_script_builder.job_script_path(job_name)
        )

        if isinstance(job, array_job_lib.ArrayJob):
            num_jobs = len(job.env_vars)
        else:
            num_jobs = 1

        console.info(f"Launching {num_jobs} jobs locally...")
        handles = []
        for i in range(num_jobs):

            def task(i):
                additional_env = {}
                if isinstance(job, array_job_lib.ArrayJob):
                    additional_env = {
                        LocalJobScriptBuilder.ARRAY_TASK_ID: str(
                            i + LocalJobScriptBuilder.ARRAY_TASK_OFFSET
                        ),
                    }
                subprocess.run(
                    ["bash", job_script_path],
                    env={**os.environ, **additional_env},
                )

            future = local_executor().submit(task, i)
            handles.append(LocalExecutionHandle(future))

        return handles


@functools.lru_cache()
def client() -> LocalClient:
    project = config_lib.default().project()
    local_settings = config_lib.default().local_settings()

    filesystem = fsspec.filesystem("file")
    storage_root = os.path.abspath(os.path.expanduser(local_settings.storage_root))
    artifact_store = artifacts.ArtifactStore(filesystem, storage_root, project=project)
    return LocalClient(local_settings, artifact_store)


def _local_job_predicate(job):
    if isinstance(job, xm.Job):
        return isinstance(job.executor, executors.Local)
    elif isinstance(job, array_job_lib.ArrayJob):
        return isinstance(job.executor, executors.Local)
    else:
        raise ValueError(f"Unexpected job type: {type(job)}")


async def launch(job_name: str, job):
    jobs = job_script_builder.flatten_job(job)
    jobs = [job for job in jobs if _local_job_predicate(job)]

    if not jobs:
        return []

    if len(jobs) > 1:
        raise ValueError(
            "Cannot launch a job group with multiple jobs as a single job."
        )

    return client().launch(job_name, jobs[0])
