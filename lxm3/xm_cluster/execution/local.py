import asyncio
import atexit
import concurrent.futures
import os
import subprocess
from typing import List, Optional

import fsspec
from absl import logging

from lxm3 import xm
from lxm3.xm_cluster import config as config_lib
from lxm3.xm_cluster.console import console
from lxm3.xm_cluster.execution import artifacts
from lxm3.xm_cluster.execution import gridengine

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

    local_config = config.local_config()
    storage_root = os.path.abspath(
        os.path.expanduser(local_config["storage"]["staging"])
    )
    fs = fsspec.filesystem("file")

    artifact = artifacts.LocalArtifact(fs, storage_root, project=config.project())
    job_script_path = gridengine.deploy_job_resources(artifact, jobs)

    console.print(f"Launching {len(jobs)} jobs locally...")
    handles = []
    for i in range(len(jobs)):

        def task(i):
            subprocess.run(
                ["bash", job_script_path], env={**os.environ, "SGE_TASK_ID": str(i + 1)}
            )

        future = local_executor().submit(task, i)
        handles.append(LocalExecutionHandle(future))

    return handles
