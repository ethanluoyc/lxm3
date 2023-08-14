from typing import Dict, Optional, Sequence, Union, Any

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

    # Placeholder, no effect right now
    requirements: JobRequirements = attr.Factory(JobRequirements)

    singularity_options: Optional[SingularityOptions] = None

    Spec = LocalSpec  # type: ignore


@attr.s(auto_attribs=True)
class GridEngineSpec(xm.ExecutorSpec):
    """Spec for SGE execution."""


@attr.s(auto_attribs=True)
class GridEngine(xm.Executor):
    """SGE executor."""

    # WARNING:
    # requirements are currently ignored as different SGE clusters
    # use different approaches for configurting these resources.
    # To configure, use resources for -l directives.
    # For UCL clusters, use the auto-configuration package from lxm3.contrib.
    requirements: JobRequirements = attr.Factory(JobRequirements)
    # Resources passed to qsub as -l
    resources: Dict[str, Any] = attr.Factory(dict)
    # Parallel environments in the form of --pe <name> <slots>
    parallel_environments: Dict[str, int] = attr.Factory(dict)
    # Maximum running time, -l h_rt
    walltime: Optional[Union[int, str]] = None

    # queue to submit the job to: -q
    queue: Optional[str] = None
    # If set, use -R y
    reserved: Optional[bool] = None
    # Log directory for stdout/stderr
    log_directory: Optional[str] = None
    # If False, log to separate files
    merge_output: bool = True
    shell: str = "/bin/bash"

    # -P
    project: Optional[str] = None
    # -A
    account: Optional[str] = None

    # Modules to load before running the job
    modules: Sequence[str] = attr.Factory(list)

    # -tc
    max_parallel_tasks: Optional[int] = None
    extra_directives: Sequence[str] = attr.Factory(list)
    skip_directives: Sequence[str] = attr.Factory(list)

    singularity_options: Optional[SingularityOptions] = None

    Spec = GridEngineSpec  # type: ignore
