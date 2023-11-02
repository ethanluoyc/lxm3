import re
from typing import Any, Dict, List, Optional

import attr

from lxm3 import xm


def _validate_env_vars(self: Any, attribute: Any, env_vars: Dict[str, str]) -> None:
    del self  # Unused.
    del attribute  # Unused.
    for key in env_vars.keys():
        if not re.fullmatch("[a-zA-Z_][a-zA-Z0-9_]*", key):
            raise ValueError(
                "Environment variables names must conform to "
                f"[a-zA-Z_][a-zA-Z0-9_]*. Got {key!r}."
            )


@attr.s(auto_attribs=True)
class WorkItem:
    args: xm.SequentialArgs = attr.ib(
        factory=list, converter=xm.SequentialArgs.from_collection
    )  # pytype: disable=annotation-type-mismatch
    env_vars: Dict[str, str] = attr.ib(
        converter=dict, default=attr.Factory(dict), validator=_validate_env_vars
    )


@attr.s(auto_attribs=True)
class WorkList:
    work_items: List[WorkItem] = attr.ib(factory=list)

    @classmethod
    def from_collection(cls, collection: List[Any]) -> "WorkList":
        items = []
        for item in collection:
            if isinstance(item, dict):
                items.append(WorkItem(**item))
            elif isinstance(item, WorkItem):
                items.append(item)
            else:
                raise TypeError(
                    "Expected WorkItem or dict with keys args, and env_vars, got {item!r}"
                )
        return cls(work_items=items)


@attr.s(auto_attribs=True)
class ArrayJob(xm.JobConfig):
    executable: xm.Executable
    executor: xm.Executor
    name: Optional[str] = None
    work_list: WorkList = attr.ib(converter=WorkList.from_collection, factory=WorkList)


def flatten_array_job(job: ArrayJob) -> List[xm.Job]:
    jobs = []
    for work_item in job.work_list.work_items:
        jobs.append(
            xm.Job(
                executable=job.executable,
                executor=job.executor,
                args=work_item.args,
                env_vars=work_item.env_vars,
            )
        )
    return jobs
