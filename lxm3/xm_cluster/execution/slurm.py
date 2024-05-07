import datetime
import functools
import os
import re
from typing import List, Optional

from lxm3 import xm
from lxm3.clusters import slurm
from lxm3.xm_cluster import array_job
from lxm3.xm_cluster import artifacts
from lxm3.xm_cluster import config as config_lib
from lxm3.xm_cluster import console
from lxm3.xm_cluster import executables
from lxm3.xm_cluster import executors
from lxm3.xm_cluster.execution import job_script_builder


class SlurmJobScriptBuilder(job_script_builder.JobScriptBuilder[executors.Slurm]):
    ARRAY_TASK_ID = "SLURM_ARRAY_TASK_ID"
    ARRAY_TASK_OFFSET = 1
    JOB_SCRIPT_SHEBANG = "#!/usr/bin/bash -l"
    JOB_ENV_PATTERN = "^(SLURM_)"

    @classmethod
    def _is_gpu_requested(cls, executor: executors.Slurm) -> bool:
        del executor
        return True  # TODO

    @classmethod
    def _create_job_script_prologue(cls, executable, executor: executors.Slurm) -> str:
        cmds = ['echo >&2 "INFO[$(basename "$0")]: Running on host $(hostname)"']

        for module in executor.modules:
            cmds.append(f"module load {module}")
        if cls._is_gpu_requested(executor):
            cmds.append(
                'echo >&2 "INFO[$(basename "$0")]: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"'
            )

        return "\n".join(cmds)

    @classmethod
    def _create_job_script_header(
        cls,
        executor: executors.Slurm,
        num_array_tasks: Optional[int],
        job_log_dir: str,
        job_name: str,
    ) -> str:
        job_header = header_from_executor(
            job_name, executor, num_array_tasks, job_log_dir
        )
        return job_header

    def build(
        self, job: job_script_builder.JobType, job_name: str, job_log_dir: str
    ) -> str:
        assert isinstance(job.executor, executors.Slurm)
        assert isinstance(job.executable, executables.AppBundle)
        return super().build(job, job_name, job_log_dir)


class SlurmHandle:
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id


class SlurmClient:
    builder_cls: type[SlurmJobScriptBuilder] = SlurmJobScriptBuilder

    def __init__(
        self,
        settings: config_lib.ClusterSettings,
        artifact_store: artifacts.ArtifactStore,
    ) -> None:
        self._settings = settings
        self._artifact_store = artifact_store

        self._cluster = slurm.SlurmCluster(
            hostname=self._settings.hostname, username=self._settings.user
        )

    @property
    def artifact_store(self):
        return self._artifact_store

    def launch(self, job_name: str, job: job_script_builder.JobType):
        job_name = re.sub("\\W", "_", job_name)
        job_log_dir = job_script_builder.job_log_path(job_name)
        self._artifact_store.ensure_dir(job_log_dir)
        job_log_dir = self._artifact_store.normalize_path(job_log_dir)

        builder = self.builder_cls(self._settings)
        job_script_content = builder.build(job, job_name, job_log_dir)
        job_script_path = self._artifact_store.put_text(
            job_script_content, job_script_builder.job_script_path(job_name)
        )

        if isinstance(job, array_job.ArrayJob):
            num_jobs = len(job.env_vars)
        else:
            num_jobs = 1
        console.info(f"Launching {num_jobs} job on {self._settings.hostname}")
        job_id = self._cluster.launch(job_script_path)
        console.info(f"Successfully launched job {job_id}")
        self._artifact_store.put_text(str(job_id), f"jobs/{job_name}/job_id")

        handles = [SlurmHandle(job_id)]

        return handles


@functools.lru_cache()
def client() -> SlurmClient:
    project = config_lib.default().project()
    settings = config_lib.default().cluster_settings()
    artifact_store = job_script_builder.create_artifact_store(settings, project)
    return SlurmClient(settings, artifact_store)


def _slurm_job_predicate(job):
    if isinstance(job, xm.Job):
        return isinstance(job.executor, executors.Slurm)
    elif isinstance(job, array_job.ArrayJob):
        return isinstance(job.executor, executors.Slurm)
    else:
        raise ValueError(f"Unexpected job type: {type(job)}")


async def launch(job_name: str, job) -> List[SlurmHandle]:
    jobs = job_script_builder.flatten_job(job)
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
    job_log_dir: str,
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

    log_directory = executor.log_directory or job_log_dir
    if num_array_tasks is not None:
        stdout = os.path.join(log_directory, "%x-%A_%a.out")
    else:
        stdout = os.path.join(log_directory, "%x-%j.out")

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
