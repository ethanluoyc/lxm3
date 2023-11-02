import datetime
from typing import Any, Dict, Optional, Sequence, Union

import attr

from lxm3 import xm
from lxm3.xm_cluster.requirements import JobRequirements


def _convert_time(
    time: Optional[Union[str, datetime.datetime, int]]
) -> Optional[datetime.timedelta]:
    if time is None:
        return None
    if isinstance(time, int):
        return datetime.timedelta(seconds=time)
    elif isinstance(time, datetime.timedelta):
        return time
    else:
        raise TypeError(
            f"Expect walltime to be type int (seconds) or datetime.timedelta, got {type(time)}"
        )


@attr.s(auto_attribs=True)
class SingularityOptions:
    """Options for singularity container.

    Args:
        bind: Mapping of the form ``{src: dst}``.
        User-bind path specification of the form ``-B <src>:<dst>``.
        extra_options: Extra commandline options to pass to singularity.

          When using the ``extra_options``, be aware of the following:
            1. lxm3 currently probably won't likely work with
               ``--contain``, ``--pwd`` and ``-wd``,
            2. You don't have to pass ``--nv`` for GPU jobs,
               lxm3 will do it for you.
    """

    bind: Dict[str, str] = attr.Factory(dict)
    extra_options: Sequence[str] = attr.Factory(list)


@attr.s(auto_attribs=True)
class DockerOptions:
    """Options for singularity container.

    Args:
        bind: Mapping of the form ``{src: dst}``.
        User-bind path specification of the form ``-B <src>:<dst>``.
        extra_options: Extra commandline options to pass to singularity.

          When using the ``extra_options``, be aware of the following:
            1. lxm3 currently probably won't likely work with
               ``--contain``, ``--pwd`` and ``-wd``,
            2. You don't have to pass ``--nv`` for GPU jobs,
               lxm3 will do it for you.
    """

    volumes: Dict[str, str] = attr.Factory(dict)
    extra_options: Sequence[str] = attr.Factory(list)


@attr.s(auto_attribs=True)
class LocalSpec(xm.ExecutorSpec):
    """Spec for local execution."""


@attr.s(auto_attribs=True)
class Local(xm.Executor):
    """Local executor.

    Args:
        requirements: placeholder, no effect right now
        singularity_options: Options for singularity container
    """

    requirements: JobRequirements = attr.Factory(JobRequirements)

    singularity_options: Optional[SingularityOptions] = None
    docker_options: Optional[DockerOptions] = None

    @classmethod
    def Spec(cls) -> LocalSpec:
        return LocalSpec()


@attr.s(auto_attribs=True)
class GridEngineSpec(xm.ExecutorSpec):
    """Spec for SGE execution."""


@attr.s(auto_attribs=True)
class GridEngine(xm.Executor):
    """SGE executor.

    Attributes:
        requirements: placeholder, no effect right now.
        resources: Resources passed to qsub as `-l key=value`.
        parallel_environments: Parallel environments in the form of ``--pe <name> <slots>``.
        walltime: Maximum running time, ``-l h_rt=time``.
            When an ``int`` is used, this is interpreted as seconds.
            A ``datetime.timedelta`` can also be used.
        queue: queue to submit the job to: ``-q``.
        reserved: If set, use ``-R y``.
        log_directory: Log directory for stdout/stderr.
        merge_output: If False, log to separate files.
        shell: Shell to use, default ``/bin/bash``.
        project: ``-P``.
        account: ``-A``.
        modules: Modules to load before running the job.
            See https://modules.readthedocs.io/en/latest/
        max_parallel_tasks: ``-tc``.
        extra_directives: Extra directives to pass to ``qsub``.
        skip_directives: Directives to skip.
        singularity_options: Options for singularity container.

    """

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
    walltime: Optional[datetime.timedelta] = attr.field(
        default=None, converter=_convert_time
    )

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
    docker_options: Optional[DockerOptions] = None

    @classmethod
    def Spec(cls) -> GridEngineSpec:
        return GridEngineSpec()


@attr.s(auto_attribs=True)
class SlurmSpec(xm.ExecutorSpec):
    """Spec for Slurm execution."""


@attr.s(auto_attribs=True)
class Slurm(xm.Executor):
    """Slurm executor."""

    requirements: JobRequirements = attr.Factory(JobRequirements)
    resources: Dict[str, Any] = attr.Factory(dict)
    walltime: Optional[datetime.timedelta] = attr.field(
        default=None, converter=_convert_time
    )

    singularity_options: Optional[SingularityOptions] = None
    docker_options: Optional[DockerOptions] = None

    log_directory: Optional[str] = None
    # Modules to load before running the job
    modules: Sequence[str] = attr.Factory(list)

    exclusive: bool = False
    partition: Optional[str] = None

    extra_directives: Sequence[str] = attr.Factory(list)
    skip_directives: Sequence[str] = attr.Factory(list)

    @classmethod
    def Spec(cls) -> SlurmSpec:
        return SlurmSpec()
