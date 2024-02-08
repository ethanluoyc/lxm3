import functools
import os
import re
from typing import List, Optional

from absl import logging

from lxm3 import xm
from lxm3.clusters import gridengine
from lxm3.xm_cluster import array_job
from lxm3.xm_cluster import artifacts
from lxm3.xm_cluster import config as config_lib
from lxm3.xm_cluster import executables
from lxm3.xm_cluster import executors
from lxm3.xm_cluster import requirements as cluster_requirements
from lxm3.xm_cluster.console import console
from lxm3.xm_cluster.execution import job_script_builder


class GridEngineJobScriptBuilder(
    job_script_builder.JobScriptBuilder[executors.GridEngine]
):
    TASK_OFFSET = 1
    JOB_SCRIPT_SHEBANG = "#!/usr/bin/env bash"
    TASK_ID_VAR_NAME = "SGE_TASK_ID"
    JOB_ENV_PATTERN = "^(JOB_|SGE_|PE|NSLOTS|NHOSTS)"

    executable_cls = executables.Command
    executor_cls = executors.GridEngine

    @classmethod
    def _is_gpu_requested(cls, executor: executors.GridEngine) -> bool:
        return (
            "gpu" in executor.parallel_environments
            or cluster_requirements.ResourceType.GPU
            in executor.requirements.task_requirements
            or "gpu" in executor.resources
        )

    @classmethod
    def _create_setup_cmds(
        cls, executable: executables.Command, executor: executors.GridEngine
    ) -> str:
        cmds = ["echo >&2 INFO[$(basename $0)]: Running on host $(hostname)"]

        for module in executor.modules:
            cmds.append(f"module load {module}")
        if cls._is_gpu_requested(executor):
            cmds.append(
                "echo >&2 INFO[$(basename $0)]: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
            )
            cmds.append("nvidia-smi")

        if executable.singularity_image is not None:
            cmds.append(
                "echo >&2 INFO[$(basename $0)]: Singularity version: $(singularity --version)"
            )
        return "\n".join(cmds)

    @classmethod
    def _create_job_header(
        cls,
        executor: executors.GridEngine,
        num_array_tasks: Optional[int],
        job_log_dir: str,
        job_name: str,
    ) -> str:
        job_header = header_from_executor(
            job_name, executor, num_array_tasks, job_log_dir
        )
        return job_header

    def build(
        self, job: job_script_builder.ClusterJob, job_name: str, job_log_dir: str
    ) -> str:
        assert isinstance(job.executor, self.executor_cls)
        assert isinstance(job.executable, self.executable_cls)
        return super().build(job, job_name, job_log_dir)


class GridEngineHandle:
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id

    async def wait(self):
        raise NotImplementedError()

    async def monitor(self):
        raise NotImplementedError()


def _sge_job_predicate(job):
    if isinstance(job, xm.Job):
        return isinstance(job.executor, executors.GridEngine)
    elif isinstance(job, array_job.ArrayJob):
        return isinstance(job.executor, executors.GridEngine)
    else:
        raise ValueError(f"Unexpected job type: {type(job)}")


class GridEngineClient:
    builder_cls = GridEngineJobScriptBuilder
    _settings: config_lib.ClusterSettings

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

        self._cluster = gridengine.GridEngineCluster(
            self._settings.hostname, self._settings.user
        )

    def launch(self, job_name: str, job: job_script_builder.ClusterJob):
        job_name = re.sub("\\W", "_", job_name)
        job_script_dir = self._artifact_store.job_path(job_name)
        job_log_dir = self._artifact_store.job_log_path(job_name)
        builder = self.builder_cls(self._settings)
        job_script_content = builder.build(job, job_name, job_log_dir)

        self._artifact_store.deploy_job_scripts(job_name, job_script_content)
        job_script_path = os.path.join(
            job_script_dir, job_script_builder.JOB_SCRIPT_NAME
        )

        if isinstance(job, array_job.ArrayJob):
            num_jobs = len(job.env_vars)
        else:
            num_jobs = 1

        console.log(f"Launching {num_jobs} job on {self._settings.hostname}")
        console.log(f"Launch with command:\n  qsub {job_script_path}")
        group = self._cluster.launch(job_script_path)
        console.log(f"Successfully launched job {group.group(0)}")
        console.log(f"Logs are saved in {job_log_dir}")

        handles = [
            GridEngineHandle(job_id) for job_id in gridengine.split_job_ids(group)
        ]

        self._save_job_id(job_script_path, group.group(0))

        return handles

    def _save_job_id(self, job_script_path: str, job_id: str):
        self._artifact_store._fs.write_text(
            os.path.join(os.path.dirname(job_script_path), "job_id"), f"{job_id}\n"
        )


@functools.lru_cache()
def client():
    return GridEngineClient()


async def launch(
    job_name: str, job: job_script_builder.ClusterJob
) -> List[GridEngineHandle]:
    if isinstance(job, array_job.ArrayJob):
        jobs = [job]  # type: ignore
    elif isinstance(job, xm.JobGroup):
        jobs: List[xm.Job] = xm.job_operators.flatten_jobs(job)
    else:
        raise NotImplementedError()

    jobs = [job for job in jobs if _sge_job_predicate(job)]

    if not jobs:
        return []

    if len(jobs) > 1:
        raise ValueError(
            "Cannot launch a job group with multiple jobs as a single job."
        )

    return client().launch(job_name, jobs[0])


def _format_time(duration_seconds: int) -> str:
    hours, remainder = divmod(duration_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))


def header_from_executor(
    job_name: str,
    executor: executors.GridEngine,
    num_array_tasks: Optional[int],
    job_log_dir: str,
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

    log_directory = executor.log_directory or job_log_dir
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
