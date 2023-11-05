import datetime
import functools
import os
from typing import List, Optional

from lxm3 import xm
from lxm3.clusters import slurm
from lxm3.xm_cluster import array_job
from lxm3.xm_cluster import artifacts
from lxm3.xm_cluster import config as config_lib
from lxm3.xm_cluster import executables
from lxm3.xm_cluster import executors
from lxm3.xm_cluster.console import console
from lxm3.xm_cluster.execution import job_script


class SlurmJobScriptBuilder(job_script.JobScriptBuilder):
    TASK_OFFSET = 1
    JOB_SCRIPT_SHEBANG = "#!/usr/bin/bash -l"
    TASK_ID_VAR_NAME = "SLURM_ARRAY_TASK_ID"

    @classmethod
    def _is_gpu_requested(cls, executor: executors.Slurm) -> bool:
        del executor
        return True  # TODO

    @classmethod
    def _create_setup_cmds(cls, executable, executor) -> str:
        cmds = ["echo >&2 INFO[$(basename $0)]: Running on host $(hostname)"]

        for module in executor.modules:
            cmds.append(f"module load {module}")
        if cls._is_gpu_requested(executor):
            cmds.append(
                "echo >&2 INFO[$(basename $0)]: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
            )
            # cmds.append("nvidia-smi")

        if executable.singularity_image is not None:
            cmds.append(
                "echo >&2 INFO[$(basename $0)]: Singularity version: $(singularity --version)"
            )
        return "\n".join(cmds)

    @classmethod
    def _create_job_header(
        cls,
        executor: executors.Slurm,
        num_array_tasks: Optional[int],
        job_script_dir: str,
        job_name: str,
    ) -> str:
        num_array_tasks = None
        job_header = header_from_executor(
            job_name, executor, num_array_tasks, job_script_dir
        )
        return job_header

    def build(
        self, job: job_script.ClusterJob, job_name: str, job_script_dir: str
    ) -> str:
        assert isinstance(job.executor, executors.Slurm)
        assert isinstance(job.executable, executables.Command)
        return super().build(job, job_name, job_script_dir)


class SlurmHandle:
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id

    async def wait(self):
        raise NotImplementedError()

    async def monitor(self):
        raise NotImplementedError()


def _slurm_job_predicate(job):
    if isinstance(job, xm.Job):
        return isinstance(job.executor, executors.Slurm)
    elif isinstance(job, array_job.ArrayJob):
        return isinstance(job.executor, executors.Slurm)
    else:
        raise ValueError(f"Unexpected job type: {type(job)}")


class SlurmClient(job_script.JobClient):
    builder_cls = SlurmJobScriptBuilder

    def __init__(
        self,
        settings: Optional[config_lib.ClusterSettings] = None,
        artifact_store: Optional[artifacts.ArtifactStore] = None,
    ) -> None:
        if settings is None:
            settings = config_lib.default().cluster_settings()
        self._settings = settings

        if artifact_store is None:
            project = config_lib.default().project()
            artifact_store = artifacts.create_artifact_store(
                settings.storage_root,
                hostname=settings.hostname,
                user=settings.user,
                project=project,
                connect_kwargs=settings.ssh_config,
            )
        self._artifact_store = artifact_store

        self._cluster = slurm.SlurmCluster(
            hostname=self._settings.hostname,
            username=self._settings.user,
        )

    def _launch(self, job_script_path, num_jobs):
        console.log(f"Launch with command:\n  sbatch {job_script_path}")
        cluster = self._cluster
        job_id = cluster.launch(job_script_path)
        console.log(f"Successfully launched job {job_id}")
        if num_jobs > 1:
            job_ids = [f"{job_id}_{i}" for i in range(num_jobs)]
        else:
            job_ids = [f"{job_id}"]
        return job_id, [SlurmHandle(j) for j in job_ids]


@functools.lru_cache()
def client() -> SlurmClient:
    return SlurmClient()


async def launch(job_name: str, job: job_script.ClusterJob) -> List[SlurmHandle]:
    if isinstance(job, array_job.ArrayJob):
        jobs = [job]  # type: ignore
    elif isinstance(job, xm.JobGroup):
        jobs: List[xm.Job] = xm.job_operators.flatten_jobs(job)
    elif isinstance(job, xm.Job):
        jobs = [job]

    jobs = [job for job in jobs if _slurm_job_predicate(job)]

    if not jobs:
        return []

    if len(jobs) > 1:
        raise ValueError(
            "Cannot launch a job group with multiple jobs as a single job."
        )

    if not isinstance(jobs[0].executor, executors.Slurm):
        raise ValueError(
            "Only GridEngine executors are supported by the gridengine backend."
        )

    return client().launch(job_name, jobs[0])


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


def header_from_executor(
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
