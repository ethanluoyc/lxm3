import datetime
import os
from typing import List, Optional

from absl import logging

from lxm3 import xm
from lxm3.clusters import gridengine
from lxm3.xm_cluster import artifacts
from lxm3.xm_cluster import config as config_lib
from lxm3.xm_cluster import executables
from lxm3.xm_cluster import executors
from lxm3.xm_cluster import requirements as cluster_requirements
from lxm3.xm_cluster.console import console
from lxm3.xm_cluster.execution import common
from lxm3.xm_cluster.execution import job_script

_TASK_OFFSET = 1
_JOB_SCRIPT_SHEBANG = "#!/usr/bin/env bash"
_TASK_ID_VAR_NAME = "SGE_TASK_ID"


def _format_time(duration_seconds: int) -> str:
    hours, remainder = divmod(duration_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))


def _generate_header_from_executor(
    job_name: str,
    executor: executors.GridEngine,
    num_array_tasks: Optional[int],
    job_script_dir: str,
) -> str:
    header = []

    header.append(f"#$ -N {job_name}")

    if executor.requirements.task_requirements:
        logging.warning(
            "Task requirements in GridEngine will be ignored."
            "Please use GridEngine.resources for customizing resources."
        )

    for resource, value in executor.resources.items():
        if value:
            header.append(f"#$ -l {resource}={value}")

    for pe_name, value in executor.parallel_environments.items():
        if value > 0:
            header.append(f"#$ -pe {pe_name} {value}")

    if executor.walltime is not None:
        duration_secs = int(executor.walltime.total_seconds())
        header.append(f"#$ -l h_rt={_format_time(duration_secs)}")

    if executor.queue:
        header.append(f"#$ -q {executor.queue}")

    reserved = executor.reserved
    if reserved is None:
        if (
            executor.parallel_environments
            or cluster_requirements.ResourceType.GPU
            in executor.requirements.task_requirements
            or "gpu" in executor.resources
        ):
            reserved = True

    if reserved:
        header.append("#$ -R y")

    log_directory = executor.log_directory or os.path.join(job_script_dir, "logs")
    if num_array_tasks is not None:
        stdout = os.path.join(log_directory, "$JOB_NAME.o$JOB_ID.$TASK_ID")
        stderr = os.path.join(log_directory, "$JOB_NAME.e$JOB_ID.$TASK_ID")
    else:
        stdout = os.path.join(log_directory, "$JOB_NAME.o$JOB_ID")
        stderr = os.path.join(log_directory, "$JOB_NAME.e$JOB_ID")

    header.append(f"#$ -o {stdout}")
    header.append(f"#$ -e {stderr}")
    if executor.merge_output:
        header.append("#$ -j y")

    if num_array_tasks is not None:
        header.append(f"#$ -t 1-{num_array_tasks}")
        if executor.max_parallel_tasks:
            header.append(f"#$ -tc {executor.max_parallel_tasks}")

    # Use current working directory
    header.append("#$ -cwd")

    header.append(f"#$ -S {executor.shell}")

    if executor.project:
        header.append(f"#$ -P {executor.project}")

    if executor.account:
        header.append(f"#$ -A {executor.account}")

    # Skip requested header directives
    header = list(
        filter(
            lambda line: not any(skip in line for skip in executor.skip_directives),
            header,
        )
    )

    for line in executor.extra_directives:
        if not line.startswith("#$"):
            line = "#$ " + line
        header.append(line)

    return "\n".join(header)


def _is_gpu_requested(executor: executors.GridEngine) -> bool:
    return (
        "gpu" in executor.parallel_environments
        or cluster_requirements.ResourceType.GPU
        in executor.requirements.task_requirements
        or "gpu" in executor.resources
    )


def _create_job_header(
    executor: executors.GridEngine,
    jobs: List[xm.Job],
    job_script_dir: str,
    job_name: str,
) -> str:
    num_array_tasks = len(jobs) if len(jobs) > 1 else None
    job_header = _generate_header_from_executor(
        job_name, executor, num_array_tasks, job_script_dir
    )
    return job_header


def _get_setup_cmds(
    executable: executables.Command,
    executor: executors.GridEngine,
) -> str:
    cmds = ["echo >&2 INFO[$(basename $0)]: Running on host $(hostname)"]

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


def create_job_script(
    cluster_settings: config_lib.ClusterSettings,
    jobs: List[xm.Job],
    job_name: str,
    job_script_dir: str,
) -> str:
    executable = jobs[0].executable
    executor = jobs[0].executor

    assert isinstance(executor, executors.GridEngine)
    assert isinstance(executable, executables.Command)
    job_script.validate_same_job_configuration(jobs)

    setup = _get_setup_cmds(executable, executor)
    header = _create_job_header(executor, jobs, job_script_dir, job_name)

    return common.create_array_job(
        executable=executable,
        singularity_image=executable.singularity_image,
        singularity_options=executor.singularity_options,
        jobs=jobs,
        use_gpu=_is_gpu_requested(executor),
        job_script_shebang=_JOB_SCRIPT_SHEBANG,
        task_offset=_TASK_OFFSET,
        task_id_var_name=_TASK_ID_VAR_NAME,
        setup=setup,
        header=header,
        settings=cluster_settings,
    )


class GridEngineHandle:
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id

    async def wait(self):
        raise NotImplementedError()

    async def monitor(self):
        raise NotImplementedError()


async def launch(
    config: config_lib.Config, jobs: List[xm.Job]
) -> List[GridEngineHandle]:
    if len(jobs) < 1:
        return []

    cluster_settings = config_lib.default().cluster_settings()

    artifact = artifacts.create_artifact_store(
        cluster_settings.storage_root,
        hostname=cluster_settings.hostname,
        user=cluster_settings.user,
        project=config.project(),
        connect_kwargs=cluster_settings.ssh_config,
    )

    version = datetime.datetime.now().strftime("%Y%m%d.%H%M%S")
    job_name = f"job-{version}"
    job_script_dir = artifact.job_path(job_name)
    job_script_content = create_job_script(
        cluster_settings, jobs, job_name, job_script_dir
    )

    artifact.deploy_job_scripts(job_name, job_script_content)
    job_script_path = os.path.join(job_script_dir, job_script.JOB_SCRIPT_NAME)

    client = gridengine.Client(cluster_settings.hostname, cluster_settings.user)
    console.log(f"Launching {len(jobs)} jobs on {cluster_settings.hostname}")

    console.log(f"Launch with command:\n  qsub {job_script_path}")
    group = client.launch(job_script_path)
    common.write_job_id(artifact, job_script_path, group.group(0))
    console.log(f"Successfully launched job {group.group(0)}")
    job_ids = gridengine.split_job_ids(group)

    return [GridEngineHandle(job_id) for job_id in job_ids]
