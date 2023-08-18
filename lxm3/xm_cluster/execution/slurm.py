# noqa
import datetime
import os
import shlex
import shutil
from typing import Any, Dict, List, Optional, Sequence, Union

from lxm3 import xm
from lxm3._vendor.xmanager.xm import job_blocks
from lxm3.clusters import slurm
from lxm3.xm_cluster import config as config_lib
from lxm3.xm_cluster import executables
from lxm3.xm_cluster import executors
from lxm3.xm_cluster.console import console
from lxm3.xm_cluster.execution import artifacts

_TASK_OFFSET = 1

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


def _get_cmd_str(cmd: Union[str, Sequence[str]]) -> str:
    if isinstance(cmd, str):
        return cmd
    else:
        return " ".join(list(map(shlex.quote, cmd)))


def create_array_wrapper_script(
    cmd: Union[str, Sequence[str]],
    work_list: List[_WorkItem],
    task_offset: int,
    shebang="#!/usr/bin/env bash",
) -> str:
    """Create a wrapper script for running parameter sweep."""
    env_vars = _create_env_vars([work.get("env_vars", {}) for work in work_list])
    args = _create_args([work.get("args", []) for work in work_list])
    script_args = {
        "shebang": shebang,
        "task_offset": task_offset,
        "env_vars": env_vars,
        "args": args,
        "cmd": _get_cmd_str(cmd),
    }
    script = _ARRAY_WRAPPER_TEMPLATE % script_args
    return script


def _create_job_script(
    cmd: str,
    header: str,
    archives: Optional[List[str]] = None,
    setup: str = "",
) -> str:
    if archives is None:
        archives = []
    return """\
#!/usr/bin/bash -l
%(header)s

set -e

_prepare_workdir() {
    WORKDIR=$(mktemp -t -d -u lxm-workdir.XXXXX)
    echo >&2 "INFO[$(basename $0)]: Prepare work directory: $WORKDIR"
    mkdir -p $WORKDIR

    _cleanup() {
        echo >&2 "INFO[$(basename $0)]: Clean up work directory: $WORKDIR"
        rm -rf $WORKDIR
    }
    trap _cleanup EXIT

    # Extract archives
    ARCHIVES="%(archives)s"
    for ar in $ARCHIVES; do
        unzip -q -d $WORKDIR $ar
    done

}

_prepare_workdir
cd $WORKDIR

%(setup)s
%(cmd)s

""" % {
        "cmd": _get_cmd_str(cmd),
        "setup": setup,
        "archives": " ".join(archives),
        "header": header,
    }


def _generate_header_from_executor(
    job_name: str,
    executor: executors.Slurm,
    num_array_tasks: Optional[int],
    job_script_dir: str,
) -> str:
    header = []

    header.append(f"#SBATCH --job-name={job_name}")
    # TODO(yl): Only one task is supported for now.
    header.append("#SBATCH --ntasks=1")

    for resource, value in executor.resources.items():
        if value:
            header.append(f"#SBATCH --{resource}={value}")

    if executor.walltime:
        if not isinstance(executor.walltime, str):
            duration = datetime.timedelta(seconds=executor.walltime)
        else:
            duration = executor.walltime
        header.append(f"#SBATCH --time={duration}")

    log_directory = executor.log_directory or os.path.join(job_script_dir, "logs")
    if num_array_tasks is not None:
        stdout = os.path.join(log_directory, "slurm-%A_%a.out")
    else:
        stdout = os.path.join(log_directory, "slurm-%j.out")

    header.append(f"#SBATCH --output={stdout}")

    if executor.exclusive:
        header.append("#SBATCH --exclusive")

    if num_array_tasks is not None:
        array_spec = f"1-{num_array_tasks}"
        header.append(f"#SBATCH --array={array_spec}")

    # Skip requested header directives
    header = list(
        filter(
            lambda line: not any(skip in line for skip in executor.skip_directives),
            header,
        )
    )

    for line in executor.extra_directives:
        if not line.startswith("#SBATCH"):
            line = "#SBATCH " + line
        header.append(line)

    return "\n".join(header)


def _is_gpu_requested(executor: executors.Slurm) -> bool:
    return True  # TODO


def _get_singulation_options(
    executor: Union[executors.Slurm, executors.Local]
) -> List[str]:
    options = executor.singularity_options or executors.SingularityOptions()
    result = []

    if options.bind is not None:
        for host, container in options.bind.items():
            result.extend(["--bind", f"{host}:{container}"])

    result.extend(list(options.extra_options))

    if isinstance(executor, executors.Slurm):
        if _is_gpu_requested(executor):
            result.append("--nv")
    elif isinstance(executor, executors.Local):
        if shutil.which("nvidia-smi"):
            result.append("--nv")
    else:
        raise ValueError(f"Unsupported executor type {type(executor)}")

    return result


def _validate_same_job_configuration(jobs: List[xm.Job]):
    executable = jobs[0].executable
    executor = jobs[0].executor
    for job in jobs[1:]:
        if job.executable != executable:
            raise ValueError("All jobs must have the same executable.")
        if job.executor != executor:
            raise ValueError("All jobs must have the same executor.")


def _create_job_header(
    executor: Union[executors.Slurm, executors.Local],
    jobs: List[xm.Job],
    job_script_dir: str,
    job_name: str,
) -> str:
    if isinstance(executor, executors.Slurm):
        num_array_tasks = len(jobs) if len(jobs) > 1 else None
        job_header = _generate_header_from_executor(
            job_name, executor, num_array_tasks, job_script_dir
        )

    elif isinstance(executor, executors.Local):
        job_header = ""
    else:
        raise TypeError(f"Unsupported executor type {type(executor)}")

    return job_header


def _create_array_wrapper(executable: executables.Command, jobs: List[xm.Job]):
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

    return create_array_wrapper_script(
        cmd=executable.entrypoint_command,
        work_list=work_list,
        task_offset=_TASK_OFFSET,
    )


def _get_setup_cmds(
    executable: executables.Command,
    executor: Union[executors.Slurm, executors.Local],
) -> str:
    cmds = ["echo >&2 INFO[$(basename $0)]: Running on host $(hostname)"]

    if isinstance(executor, executors.Slurm):
        for module in executor.modules:
            cmds.append(f"module load {module}")
        if _is_gpu_requested(executor):
            cmds.append(
                "echo >&2 INFO[$(basename $0)]: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
            )
            cmds.append("nvidia-smi")

    if executable.singularity_image is not None:
        cmds.append(
            "echo >&2 INFO[$(basename $0)]: Singularity version: $(singularity --version)"
        )
    return "\n".join(cmds)


def deploy_job_resources(
    artifact: artifacts.Artifact,
    jobs: List[xm.Job],
    version: Optional[str] = None,
) -> str:
    version = version or datetime.datetime.now().strftime("%Y%m%d.%H%M%S")

    executable = jobs[0].executable
    executor = jobs[0].executor

    assert isinstance(executor, executors.Slurm) or isinstance(
        executor, executors.Local
    )
    assert isinstance(executable, executables.Command)
    _validate_same_job_configuration(jobs)

    job_name = f"job-{version}"

    job_script_dir = artifact.job_path(job_name)

    deploy_array_wrapper_path = os.path.join(job_script_dir, "array_wrapper.sh")
    # Always use the array wrapper for but hardcode task id if not launching one job.
    if len(jobs) > 1:
        job_command = " ".join(
            ["sh", os.fspath(deploy_array_wrapper_path), "$SLURM_ARRAY_TASK_ID"]
        )
    else:
        job_command = " ".join(["sh", os.fspath(deploy_array_wrapper_path), "1"])

    singularity_image = executable.singularity_image
    if singularity_image is not None:
        deploy_container_path = artifact.singularity_image_path(
            os.path.basename(singularity_image)
        )

        singularity_opts = " ".join(_get_singulation_options(executor))
        job_command = (
            f"singularity exec {singularity_opts} {deploy_container_path} {job_command}"
        )

    array_wrapper = _create_array_wrapper(executable, jobs)
    deploy_archive_path = artifact.archive_path(executable.resource_uri)
    deploy_job_script_path = os.path.join(job_script_dir, "job.sh")
    job_script = _create_job_script(
        job_command,
        _create_job_header(executor, jobs, job_script_dir, job_name),
        archives=[deploy_archive_path],
        setup=_get_setup_cmds(executable, executor),
    )

    # Put artifacts on the staging fs
    artifact.deploy_resource_archive(executable.resource_uri)

    if singularity_image is not None:
        artifact.deploy_singularity_container(singularity_image)

    artifact.deploy_job_scripts(job_name, job_script, array_wrapper)

    return deploy_job_script_path


class SlurmHandle:
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id

    async def wait(self):
        raise NotImplementedError()

    async def monitor(self):
        raise NotImplementedError()


async def launch(config: config_lib.Config, jobs: List[xm.Job]) -> List[SlurmHandle]:
    if len(jobs) < 1:
        return []

    if not isinstance(jobs[0].executor, executors.Slurm):
        raise ValueError(
            "Only GridEngine executors are supported by the gridengine backend."
        )

    location = jobs[0].executor.requirements.location
    for job in jobs[1:]:
        executor = job.executor
        if not isinstance(executor, executors.Slurm):
            raise ValueError(
                "Only GridEngine executors are supported by the gridengine backend."
            )
        if executor.requirements.location != location:
            raise ValueError("All jobs must be launched on the same cluster.")

    if location is None:
        location = config.default_cluster()

    cluster_config = config.cluster_config(location)
    storage_root = cluster_config["storage"]["staging"]
    hostname = cluster_config.get("server", None)
    user = cluster_config.get("user", None)

    if hostname is None:
        artifact = artifacts.LocalArtifact(storage_root, project=config.project())
    else:
        artifact = artifacts.RemoteArtifact(
            hostname, user, storage_root, config.project()
        )

    job_script_path = deploy_job_resources(artifact, jobs)

    console.log(f"Launch with command:\n  sbatch {job_script_path}")
    client = slurm.Client(hostname=hostname, username=user)

    job_id = client.launch(job_script_path)
    artifact._fs.write_text(
        os.path.join(os.path.dirname(job_script_path), "job_id"), f"{job_id}\n"
    )
    if len(jobs) > 1:
        job_ids = [f"{job_id}_{i}" for i in range(len(jobs))]
    else:
        job_ids = [f"{job_id}"]
    console.log(f"Successfully launched job {job_id}")

    return [SlurmHandle(j) for j in job_ids]
