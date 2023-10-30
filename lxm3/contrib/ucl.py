"""Utilities for automatically setting up UCL GridEngine executors.
This module includes utility for UCL users to automatically set up the
cluster-specific job requirements given more abstract requirements.

This module currently only works with the CS HPC cluster and myriad.
To use this, you need to set up the configuration file such that it includes
a cluster named "cs" and a cluster named "myriad".
An example configuration looks like the following:

```
project = ""
[local.storage]
staging = "/home/<your username>/.cache/lxm3"

[[clusters]]
name = "cs"
# Or any other login node.
server = "beaker.cs.ucl.ac.uk"
user = "<redacted>"

[clusters.storage]
staging = "/home/<your user name>/lxm3-staging"

[[clusters]]
name = "myriad"
server = "myriad.cs.ucl.ac.uk"
user = "<redacted>"

# staging must be a directory that is accessible from both the login node
# and compute node. For myriad, it is not correct to use
/home/<user>/Scratch as that is a symlink only visible from the login node.
[clusters.storage]
staging = "/scratch/scratch/<your user name>/lxm3-staging"

```

Then in your launcher script you can use something like the following:
```
requirements = xm_cluster.JobRequirements(gpu=1, ram=8 * xm.GB)
executor = ucl.UclGridEngine(requirements, walltime = 8 * xm.Hr)
```

This translates to the correct way to specify job resources '-l' for cs and myriad.
If you have configured both clusters, you can choose which cluster to use by
either setting requirements.location or using $LXM_CLUSTER environment variable.
If both are not set, the default is the first cluster in the configuration file.

To further customize the executor, you can manually modifer `executor.resources`
and `executor.parallel_environments` after the call to `ucl.UclGridEngine`.

To understand the rules for translating for the two clusters, refer to `:ucl_test`.

"""
from typing import Optional

from lxm3.xm_cluster import config
from lxm3.xm_cluster import executors as cluster_executors
from lxm3.xm_cluster import requirements as cluster_requirements


def _myriad_executor_fn(requirements: cluster_requirements.JobRequirements, **kwargs):
    resources = {}
    parallel_environments = {}
    for resource, value in requirements.task_requirements.items():
        if resource == cluster_requirements.ResourceType.RAM:
            resources["mem"] = value
        elif resource == cluster_requirements.ResourceType.GPU and value > 0:
            resources["gpu"] = value
        elif resource == cluster_requirements.ResourceType.CPU and value > 1:
            parallel_environments["smp"] = value

    return cluster_executors.GridEngine(
        resources=resources,
        parallel_environments=parallel_environments,
        modules=["singularity-env"],
        **kwargs,
    )


def _cs_executor_fn(requirements: cluster_requirements.JobRequirements, **kwargs):
    resources = {}
    parallel_environments = {}
    for resource, value in requirements.task_requirements.items():
        if resource == cluster_requirements.ResourceType.RAM:
            resources["h_vmem"] = value
            resources["tmem"] = value
        elif resource == cluster_requirements.ResourceType.GPU and value > 0:
            resources["gpu"] = "true"
            if value > 1:
                parallel_environments["gpu"] = value
        elif resource == cluster_requirements.ResourceType.CPU and value > 1:
            parallel_environments["smp"] = value

    if requirements.task_requirements.get(cluster_requirements.ResourceType.GPU, 0) > 0:
        resources.pop("h_vmem", None)

    return cluster_executors.GridEngine(
        resources=resources,
        parallel_environments=parallel_environments,
        **kwargs,
    )


def UclGridEngine(
    requirements: cluster_requirements.JobRequirements,
    walltime: Optional[int] = None,
    **kwargs,
) -> cluster_executors.GridEngine:
    """Create a UCL speicfic GridEngine executor.
    Args:
        requirements: The job requirements.
        walltime: The walltime in seconds.
        **kwargs: Additional arguments to pass to the executor.
        See xm_cluster.GridEngine.
    """
    executor_fns = {
        "myriad": _myriad_executor_fn,
        "cs": _cs_executor_fn,
    }
    if requirements.location is not None:
        raise ValueError("location is not supported requirements")
    cluster = config.default().default_cluster()
    if cluster not in executor_fns:
        raise ValueError(
            f"Unsupported location {cluster} for UCL GridEngine. Supported locations: "
            f"{list(executor_fns.keys())}"
        )
    executor = executor_fns[cluster](requirements, walltime=walltime, **kwargs)
    return executor
