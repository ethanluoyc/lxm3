# noqa
import datetime
import os
from typing import List, Optional

from lxm3 import xm
from lxm3.clusters import slurm
from lxm3.xm_cluster import config as config_lib
from lxm3.xm_cluster import executables
from lxm3.xm_cluster import executors
from lxm3.xm_cluster.console import console
from lxm3.xm_cluster.execution import artifacts
from lxm3.xm_cluster.execution import common
from lxm3.xm_cluster.execution import job_script

_TASK_OFFSET = 1
_JOB_SCRIPT_SHEBANG = "#!/usr/bin/bash -l"
_TASK_ID_VAR_NAME = "SLURM_ARRAY_TASK_ID"


def _format_slurm_time(duration: datetime.timedelta) -> str:
    # See
    # https://github.com/SchedMD/slurm/blob/master/src/common/parse_time.c#L786
    days = duration.days
    seconds = int(duration.seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days > 0:
        return "{:02}-{:02}:{:02}:{:02}".format(duration.days, hours, minutes, seconds)
    else:
        return "{:02}:{:02}:{:02}".format(hours, minutes, seconds)


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

    if executor.walltime is not None:
        duration = executor.walltime
        header.append(f"#SBATCH --time={_format_slurm_time(duration)}")

    log_directory = executor.log_directory or os.path.join(job_script_dir, "logs")
    if num_array_tasks is not None:
        stdout = os.path.join(log_directory, "slurm-%A_%a.out")
    else:
        stdout = os.path.join(log_directory, "slurm-%j.out")

    header.append(f"#SBATCH --output={stdout}")

    if executor.exclusive:
        header.append("#SBATCH --exclusive")

    if executor.partition:
        header.append(f"#SBATCH --partition={executor.partition}")

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


def _create_job_header(
    executor: executors.Slurm, jobs: List[xm.Job], job_script_dir: str, job_name: str
) -> str:
    num_array_tasks = len(jobs) if len(jobs) > 1 else None
    job_header = _generate_header_from_executor(
        job_name, executor, num_array_tasks, job_script_dir
    )

    return job_header


def _get_setup_cmds(
    executable: executables.Command,
    executor: executors.Slurm,
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


def deploy_job_resources(
    artifact: artifacts.Artifact,
    jobs: List[xm.Job],
    version: Optional[str] = None,
) -> str:
    version = version or datetime.datetime.now().strftime("%Y%m%d.%H%M%S")

    executable = jobs[0].executable
    executor = jobs[0].executor

    assert isinstance(executor, executors.Slurm)
    assert isinstance(executable, executables.Command)
    job_script.validate_same_job_configuration(jobs)

    job_name = f"job-{version}"

    job_script_dir = artifact.job_path(job_name)

    setup = _get_setup_cmds(executable, executor)
    header = _create_job_header(executor, jobs, job_script_dir, job_name)

    return common.create_array_job(
        artifact=artifact,
        executable=executable,
        singularity_image=executable.singularity_image,
        singularity_options=executor.singularity_options,
        jobs=jobs,
        use_gpu=_is_gpu_requested(executor),
        version=version,
        job_script_shebang=_JOB_SCRIPT_SHEBANG,
        task_offset=_TASK_OFFSET,
        task_id_var_name=_TASK_ID_VAR_NAME,
        setup=setup,
        header=header,
    )


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

    storage_root, hostname, user = common.get_cluster_settings(config, jobs)

    artifact = artifacts.create_artifact_store(
        storage_root, hostname=hostname, user=user, project=config.project()
    )

    job_script_path = deploy_job_resources(artifact, jobs)

    console.log(f"Launch with command:\n  sbatch {job_script_path}")
    client = slurm.Client(hostname=hostname, username=user)

    job_id = client.launch(job_script_path)
    common.write_job_id(artifact, job_script_path, str(job_id))
    if len(jobs) > 1:
        job_ids = [f"{job_id}_{i}" for i in range(len(jobs))]
    else:
        job_ids = [f"{job_id}"]
    console.log(f"Successfully launched job {job_id}")

    return [SlurmHandle(j) for j in job_ids]
