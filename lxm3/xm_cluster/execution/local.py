import asyncio
import atexit
import concurrent.futures
import os
import subprocess
from typing import List

import termcolor
from absl import logging

from lxm3 import xm
from lxm3.xm_cluster.execution import gridengine

_LOCAL_EXECUTOR = None


def local_executor():
    global _LOCAL_EXECUTOR
    if _LOCAL_EXECUTOR is None:
        _LOCAL_EXECUTOR = concurrent.futures.ThreadPoolExecutor(1)

        def shutdown():
            logging.debug("Shutting down local executor...")
            _LOCAL_EXECUTOR.shutdown()

        atexit.register(shutdown)
    return _LOCAL_EXECUTOR


class LocalExecutionHandle:
    def __init__(self, future) -> None:
        self.future = future

    async def wait(self):
        return await asyncio.wrap_future(self.future)

    async def monitor(self):
        await asyncio.wrap_future(self.future)


async def launch(fs, storage_root, jobs: List[xm.Job]):
    if len(jobs) < 1:
        return []

    job_script_path = gridengine.deploy_job_resources(fs, storage_root, jobs)

    print(termcolor.colored(f"Launching {len(jobs)} jobs locally...", "cyan"))
    handles = []
    for i in range(len(jobs)):

        def task(i):
            subprocess.run(
                ["bash", job_script_path], env={**os.environ, "SGE_TASK_ID": str(i + 1)}
            )

        future = local_executor().submit(task, i)
        handles.append(LocalExecutionHandle(future))

    return handles
