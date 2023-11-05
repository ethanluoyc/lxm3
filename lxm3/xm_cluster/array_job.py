import copy
from typing import Dict, List, Optional

import attr

from lxm3 import xm


@attr.s(auto_attribs=True)
class ArrayJob(xm.JobConfig):
    executable: xm.Executable
    executor: xm.Executor
    name: Optional[str] = None
    args: List[xm.SequentialArgs] = attr.ib(
        converter=lambda args_list: list(
            map(xm.SequentialArgs.from_collection, args_list)
        ),
        default=attr.Factory(lambda: [xm.SequentialArgs()]),
    )
    env_vars: List[Dict[str, str]] = attr.ib(default=attr.Factory(lambda: [{}]))

    def __attrs_post_init__(self):
        self.args, self.env_vars = _broadcast_args(self.args, self.env_vars)


def _broadcast_args(args, env_vars):
    if len(args) == len(env_vars):
        return args, env_vars
    broadcast_size = max(len(args), len(env_vars))
    if min(len(args), len(env_vars)) != 1:
        raise ValueError(
            "args and env_vars must be the same length can be broadcast to same length"
        )
    if len(args) == 1:
        args = [copy.deepcopy(args[0]) for _ in range(broadcast_size)]
    if len(env_vars) == 1:
        env_vars = [copy.deepcopy(env_vars[0]) for _ in range(broadcast_size)]

    return args, env_vars


def flatten_array_job(job: ArrayJob) -> List[xm.Job]:
    jobs = []
    assert len(job.args) == len(job.env_vars)
    for args, env_vars in zip(job.args, job.env_vars):
        jobs.append(
            xm.Job(
                executable=job.executable,
                executor=job.executor,
                args=args,
                env_vars=env_vars,
            )
        )
    return jobs
