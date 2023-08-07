from typing import Dict, Optional, Sequence, Union

import attr

from lxm3 import xm
from lxm3.xm import utils
from lxm3.xm_cluster.requirements import JobRequirements


@attr.s(auto_attribs=True)
class LocalSpec(xm.ExecutorSpec):
    """Spec for local execution."""


@attr.s(auto_attribs=True)
class Local(xm.Executor):
    """Local executor."""

    requirements: JobRequirements = attr.Factory(JobRequirements)

    singularity_container: Optional[str] = None
    singularity_options: Sequence[str] = attr.Factory(list)

    Spec = LocalSpec  # type: ignore

    def __attrs_post_init__(self):
        if self.singularity_container is not None:
            self.singularity_container = utils.resolve_path_relative_to_launcher(
                self.singularity_container
            )


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

    max_parallel_tasks: Optional[int] = None
    extra_directives: Sequence[str] = attr.Factory(list)
    skip_directives: Sequence[str] = attr.Factory(list)

    singularity_container: Optional[str] = None
    singularity_options: Sequence[str] = attr.Factory(list)

    Spec = GridEngineSpec  # type: ignore

    def __attrs_post_init__(self):
        if self.singularity_container is not None:
            self.singularity_container = utils.resolve_path_relative_to_launcher(
                self.singularity_container
            )
