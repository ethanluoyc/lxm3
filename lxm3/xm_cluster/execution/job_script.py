import shlex
from typing import Any, Dict, List, Optional, Sequence, Union

from lxm3 import xm
from lxm3._vendor.xmanager.xm import job_blocks
from lxm3.xm_cluster import executables
from lxm3.xm_cluster import executors

_DEFAULT_ARRAY_WRAPPER_SHEBANG = "#!/usr/bin/env bash"

ARRAY_WRAPPER_NAME = "array_wrapper.sh"
JOB_SCRIPT_NAME = "job.sh"

_ARRAY_WRAPPER_TEMPLATE = """\
%(shebang)s

TASK_OFFSET="%(task_offset)d"

E_NOARGS=85
if [ $# -ne "1" ]
then
  echo "Usage: `basename $0` index"
  exit $E_NOARGS
else
  TASK_ID=$(($1 - $TASK_OFFSET))
fi

# Set up environment variable for task
__array_set_env_vars() {
%(env_vars)s
}

# Print command arguments for task
__array_get_args() {
%(args)s
}

__array_set_env_vars $TASK_ID
%(cmd)s $(__array_get_args $TASK_ID)
exit $?
"""

_WorkItem = Dict[str, Any]


def _create_env_vars(env_vars_list: List[Dict[str, str]]) -> str:
    """Create the env_vars list."""
    lines = []
    first_keys = set(env_vars_list[0].keys())
    if not first_keys:
        return ":;"
    for env_vars in env_vars_list:
        if first_keys != set(env_vars.keys()):
            raise ValueError("Expect all environment variables to have the same keys")
    for key in first_keys:
        for task_id, env_vars in enumerate(env_vars_list):
            lines.append(
                '{key}_{task_id}="{value}"'.format(
                    key=key, task_id=task_id, value=env_vars[key]
                )
            )
        lines.append('{key}=$(eval echo \\$"{key}_$1")'.format(key=key))
        lines.append("export {key}".format(key=key))
    content = "\n".join(lines)
    return content


def _create_args(args_list: List[List[str]]) -> str:
    """Create the args list."""
    if not args_list:
        return ":;"
    lines = []
    for task_id, args in enumerate(args_list):
        args_str = " ".join([a for a in args])
        lines.append(
            'TASK_CMD_ARGS_{task_id}="{args_str}"'.format(
                task_id=task_id, args_str=args_str
            )
        )
    lines.append('TASK_CMD_ARGS=$(eval echo \\$"TASK_CMD_ARGS_$1")')
    lines.append("echo $TASK_CMD_ARGS")
    content = "\n".join(lines)
    return content


def worklist_from_jobs(
    executable: executables.Command, jobs: List[xm.Job]
) -> List[_WorkItem]:
    work_list = []
    for job in jobs:
        work_list.append(
            {
                "args": job_blocks.merge_args(executable.args, job.args).to_list(
                    xm.utils.ARG_ESCAPER
                ),
                "env_vars": {**executable.env_vars, **job.env_vars},
            }
        )
    return work_list


def cmd2str(cmd: Union[str, Sequence[str]]) -> str:
    if isinstance(cmd, str):
        return cmd
    else:
        return " ".join(list(map(shlex.quote, cmd)))


def create_array_wrapper_script(
    cmd: Union[str, Sequence[str]], work_list: List[_WorkItem], task_offset: int
) -> str:
    """Create a wrapper script for running parameter sweep."""
    env_vars = _create_env_vars([work.get("env_vars", {}) for work in work_list])
    args = _create_args([work.get("args", []) for work in work_list])
    script_args = {
        "shebang": _DEFAULT_ARRAY_WRAPPER_SHEBANG,
        "task_offset": task_offset,
        "env_vars": env_vars,
        "args": args,
        "cmd": cmd2str(cmd),
    }
    script = _ARRAY_WRAPPER_TEMPLATE % script_args
    return script


def create_job_script(
    cmd: str,
    header: str,
    setup: str = "",
    shebang="#!/usr/bin/bash -l",
) -> str:
    return """\
%(shebang)s
%(header)s

%(setup)s
%(cmd)s

""" % {
        "cmd": cmd2str(cmd),
        "shebang": shebang,
        "setup": setup,
        "header": header,
    }


def validate_same_job_configuration(jobs: List[xm.Job]):
    executable = jobs[0].executable
    executor = jobs[0].executor
    for job in jobs[1:]:
        if job.executable != executable:
            raise ValueError("All jobs must have the same executable.")
        if job.executor != executor:
            raise ValueError("All jobs must have the same executor.")


def get_singulation_options(
    singularity_options: Optional[executors.SingularityOptions], use_gpu: bool
) -> List[str]:
    options = singularity_options or executors.SingularityOptions()
    result = []

    if options.bind is not None:
        for host, container in options.bind.items():
            result.extend(["--bind", f"{host}:{container}"])

    result.extend(list(options.extra_options))

    if use_gpu:
        result.append("--nv")
    return result
