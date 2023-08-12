from typing import Dict, Optional, Sequence, Union

import attr

from lxm3 import xm
from lxm3.xm_cluster.requirements import JobRequirements


@attr.s(auto_attribs=True)
class SingularityOptions(xm.ExecutorSpec):
    """Options for singularity container."""

    # Dict of the form {host_path: container_path}
    bind: Optional[Dict[str, str]] = None
    # Extra commandline options to pass to singularity
    extra_options: Sequence[str] = attr.Factory(list)


@attr.s(auto_attribs=True)
class LocalSpec(xm.ExecutorSpec):
    """Spec for local execution."""


@attr.s(auto_attribs=True)
class Local(xm.Executor):
    """Local executor."""

    requirements: JobRequirements = attr.Factory(JobRequirements)

    singularity_options: Optional[SingularityOptions] = None

    Spec = LocalSpec  # type: ignore


@attr.s(auto_attribs=True)
class GridEngineSpec(xm.ExecutorSpec):
    """Spec for SGE execution."""


@attr.s(auto_attribs=True)
class GridEngine(xm.Executor):
    """SGE executor."""

    requirements: JobRequirements = attr.Factory(JobRequirements)
    parallel_environments: Dict[str, int] = attr.Factory(dict)
    walltime: Optional[Union[int, str]] = None

    queue: Optional[str] = None
    reserved: Optional[bool] = None
    log_directory: Optional[str] = None
    merge_output: bool = True
    shell: str = "/bin/bash"

    project: Optional[str] = None
    account: Optional[str] = None

    max_parallel_tasks: Optional[int] = None
    extra_directives: Sequence[str] = attr.Factory(list)
    skip_directives: Sequence[str] = attr.Factory(list)

    singularity_options: Optional[SingularityOptions] = None

    Spec = GridEngineSpec  # type: ignore
