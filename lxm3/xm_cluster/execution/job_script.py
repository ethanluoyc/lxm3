import abc
import collections
import os
import re
import shlex
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

from lxm3 import xm
from lxm3._vendor.xmanager.xm import job_blocks
from lxm3.xm_cluster import array_job
from lxm3.xm_cluster import artifacts
from lxm3.xm_cluster import config as config_lib
from lxm3.xm_cluster import executables
from lxm3.xm_cluster import executors

JOB_SCRIPT_NAME = "job.sh"
CONTAINER_WORKDIR = "/run/task"

ClusterJob = Union[xm.Job, array_job.ArrayJob]


class _WorkItem(TypedDict):
    args: List[str]
    env_vars: Dict[str, str]


ExecutorClsType = TypeVar("ExecutorClsType")


def create_singularity_entrypoint(
    cmd: List[str],
    singularity_image: str,
    singularity_options: executors.SingularityOptions,
    use_gpu: bool,
) -> str:
    opts = []

    if singularity_options.bind is not None:
        for host, container in singularity_options.bind.items():
            opts.extend(["--bind", f"{host}:{container}"])

    opts.extend(list(singularity_options.extra_options))

    if use_gpu:
        opts.append("--nv")

    return " ".join(["singularity", "exec", *opts, singularity_image, *cmd])


def create_docker_entrypoint(
    cmd: List[str],
    docker_image: str,
    docker_options: executors.DockerOptions,
    use_gpu: bool,
) -> str:
    opts = ["--rm"]
    if docker_options.volumes is not None:
        for host, container in docker_options.volumes.items():
            opts.extend(["--mount", f"type=bind,source={host},target={container}"])
    opts.extend(list(docker_options.extra_options))

    if use_gpu:
        opts.extend(
            [
                "--runtime=nvidia",
                "--env",
                '"NVIDIA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"',
            ]
        )

    return " ".join(["docker", "run", *opts, docker_image, *cmd])


class JobScriptBuilder(abc.ABC):
    TASK_OFFSET: int
    JOB_SCRIPT_SHEBANG: str = "#!/usr/bin/env bash"
    TASK_ID_VAR_NAME: str
    JOB_ENV_PATTERN = None

    def __init__(
        self,
        settings: Optional[config_lib.ExecutionSettings] = None,
    ) -> None:
        self._settings = settings

    @classmethod
    @abc.abstractmethod
    def _create_job_header(
        cls,
        executor: xm.Executor,
        num_array_tasks: Optional[int],
        job_log_dir: str,
        job_name: str,
    ) -> str:
        """Create a job header"""

    @classmethod
    @abc.abstractmethod
    def _is_gpu_requested(cls, executor: xm.Executor) -> bool:
        """Is GPU requested?"""

    @classmethod
    @abc.abstractmethod
    def _create_setup_cmds(cls, executable, executor) -> str:
        """Generate backend specific setup commands"""

    @staticmethod
    def _get_additional_env(env_vars, parent_env):
        env = {}
        for k, v in parent_env.items():
            if k not in env_vars:
                env[k] = v
        return env

    @staticmethod
    def _get_additional_binds(bind, singularity_additional_binds):
        updates = {}
        for src, dst in singularity_additional_binds.items():
            if dst not in bind.values():
                updates[src] = dst
        return updates

    @staticmethod
    def _create_env_vars(env_vars_list: List[Dict[str, str]]) -> str:
        """Create the env_vars list."""
        lines = []
        first_keys = set(env_vars_list[0].keys())
        if not first_keys:
            return ""

        # Find out keys that are common to all environment variables
        var_to_values = collections.defaultdict(list)
        for env in env_vars_list:
            for k, v in env.items():
                var_to_values[k].append(v)

        common_keys = []
        for k, v in var_to_values.items():
            if len(set(v)) == 1:
                common_keys.append(k)
        common_keys = sorted(common_keys)

        for env_vars in env_vars_list:
            if first_keys != set(env_vars.keys()):
                raise ValueError(
                    "Expect all environment variables to have the same keys"
                )

        # Generate shared environment variables
        for k in sorted(common_keys):
            lines.append(
                'export {key}="{value}"'.format(key=k, value=env_vars_list[0][k])
            )

        for key in first_keys:
            if key in common_keys:
                continue
            for task_id, env_vars in enumerate(env_vars_list):
                lines.append(
                    '{key}_{task_id}="{value}"'.format(
                        key=key, task_id=task_id, value=env_vars[key]
                    )
                )
            lines.append('{key}=$(eval echo \\$"{key}_$LXM_TASK_ID")'.format(key=key))
            lines.append("export {key}".format(key=key))
        content = "\n".join(lines)
        return content

    @staticmethod
    def _create_args(args_list: List[List[str]]) -> str:
        """Create the args list."""
        if not args_list:
            return ""
        lines = []
        for task_id, args in enumerate(args_list):
            args_str = " ".join([a for a in args])
            lines.append(
                'TASK_CMD_ARGS_{task_id}="{args_str}"'.format(
                    task_id=task_id, args_str=args_str
                )
            )
        lines.append('TASK_CMD_ARGS=$(eval echo \\$"TASK_CMD_ARGS_$LXM_TASK_ID")')
        lines.append("eval set -- $TASK_CMD_ARGS")
        content = "\n".join(lines)
        return content

    @staticmethod
    def _create_work_list(job: Union[xm.Job, array_job.ArrayJob]) -> List[_WorkItem]:
        work_list: List[_WorkItem] = []
        executable = cast(executables.Command, job.executable)
        if isinstance(job, array_job.ArrayJob):
            jobs = array_job.flatten_array_job(job)
        else:
            jobs = [job]
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

    def _create_array_script(self, job) -> str:
        executable = job.executable
        singularity_image = executable.singularity_image

        if self._settings is not None:
            env_overrides = self._settings.env
            singularity_env_overrides = self._settings.env
        else:
            env_overrides = {}
            singularity_env_overrides = {}

        work_list = self._create_work_list(job)
        for work_item in work_list:
            if singularity_image is not None:
                work_item["env_vars"].update(
                    self._get_additional_env(
                        work_item["env_vars"], singularity_env_overrides
                    )
                )
            work_item["env_vars"].update(
                self._get_additional_env(work_item["env_vars"], env_overrides)
            )
        return f"""
{self._create_env_vars([work["env_vars"] for work in work_list])}
{self._create_args([work["args"] for work in work_list])}
"""

    def _create_job_script(self, job: ClusterJob, setup: str, header: str) -> str:
        executable = job.executable
        if not isinstance(executable, executables.Command):
            raise ValueError("Only Command executable is supported")
        executor = cast(
            Union[executors.Local, executors.GridEngine, executors.Slurm], job.executor
        )
        num_tasks = len(job.args) if isinstance(job, array_job.ArrayJob) else 1
        array_script = self._create_array_script(job)

        def _rewrite_array_job_command(array_scrpt_path: str, cmd: str):
            return ["sh", "-c", shlex.quote(f'. {array_scrpt_path}; {cmd} "$@"')]

        if executable.singularity_image is not None:
            if executor.singularity_options is not None:
                singularity_options = executors.SingularityOptions(
                    bind={**executor.singularity_options.bind},
                    extra_options=[*executor.singularity_options.extra_options],
                )
            else:
                singularity_options = executors.SingularityOptions()

            if self._settings is not None:
                singularity_options.bind.update(
                    self._get_additional_binds(
                        singularity_options.bind, self._settings.singularity.binds
                    )
                )

            singularity_options.extra_options = [
                *singularity_options.extra_options,
                f'--bind="$LXM_WORKDIR:{CONTAINER_WORKDIR}"',
                f"--pwd={CONTAINER_WORKDIR}",
                '--bind="$LXM_WORKDIR/job-param.sh:/etc/job-param.sh:ro"',
                '--env=LXM_TASK_ID="$LXM_TASK_ID"',
            ]

            entrypoint = create_singularity_entrypoint(
                _rewrite_array_job_command(
                    "/etc/job-param.sh", executable.entrypoint_command
                ),
                executable.singularity_image,
                singularity_options,
                self._is_gpu_requested(executor),
            )
        elif executable.docker_image is not None:
            if executor.docker_options is not None:
                docker_options = executors.DockerOptions(
                    volumes={**executor.docker_options.volumes},
                    extra_options=[*executor.docker_options.extra_options],
                )
            else:
                docker_options = executors.DockerOptions()

            docker_options.extra_options = [
                *docker_options.extra_options,
                f'--mount="type=bind,source=$LXM_WORKDIR,target={CONTAINER_WORKDIR}"',
                f"--workdir={CONTAINER_WORKDIR}",
                '--mount="type=bind,source=$LXM_WORKDIR/job-param.sh,target=/etc/job-param.sh,readonly"',
                '--env="LXM_TASK_ID=$LXM_TASK_ID"',
            ]

            entrypoint = create_docker_entrypoint(
                _rewrite_array_job_command(
                    "/etc/job-param.sh", executable.entrypoint_command
                ),
                executable.docker_image,
                docker_options,
                self._is_gpu_requested(executor),
            )
        else:
            entrypoint = " ".join(
                _rewrite_array_job_command(
                    "./job-param.sh", executable.entrypoint_command
                )
            )

        cmds = """\
TASK_OFFSET=%(task_offset)s
TASK_INDEX_NAME="%(task_index_name)s"
NUM_TASKS="%(num_tasks)s"

if [ $NUM_TASKS -eq 1 ]; then
    # If there is only one task, then we don't need to use the task index
    LXM_TASK_ID=0
else
    LXM_TASK_ID=$(($(eval echo \\$"TASK_INDEX_NAME") - $TASK_OFFSET))
fi
export LXM_TASK_ID
cat <<'EOF' > $LXM_WORKDIR/job-param.sh
%(array_script)s
EOF
chmod +x $LXM_WORKDIR/job-param.sh
%(entrypoint)s
""" % {
            "task_offset": self.TASK_OFFSET,
            "task_index_name": self.TASK_ID_VAR_NAME,
            "num_tasks": num_tasks,
            "entrypoint": entrypoint,
            "array_script": array_script,
        }

        job_script_content = """\
%(shebang)s
%(header)s
%(setup)s
set -e

LXM_WORKDIR=$(mktemp -d -t lxm-XXXXXX)
# Extract archives
ARCHIVES="%(archives)s"
for ar in $ARCHIVES; do
    case $ar in
    *.zip)
        unzip -q -d $LXM_WORKDIR $ar
        ;;
    *.tar)
        tar -C $LXM_WORKDIR -xf $ar
        ;;
    *.tar.gz|*.tgz)
        tar -C $LXM_WORKDIR -xzf $ar
        ;;
    *)
        _error "Unsupported archive format: $ar"
        ;;
    esac
done
cleanup() {
  echo >& 2 "DEBUG[$(basename $0)] Cleaning up $LXM_WORKDIR"
  rm -rf $LXM_WORKDIR
}
trap cleanup EXIT

cd $LXM_WORKDIR

%(entrypoint)s
""" % {
            "shebang": self.JOB_SCRIPT_SHEBANG,
            "header": header,
            "setup": setup,
            "entrypoint": cmds,
            "archives": " ".join([executable.resource_uri]),
        }
        return job_script_content

    def build(
        self,
        job: Union[xm.Job, array_job.ArrayJob],
        job_name: str,
        job_log_dir: str,
    ) -> str:
        setup = self._create_setup_cmds(job.executable, job.executor)

        num_array_tasks = None
        if isinstance(job, array_job.ArrayJob) and len(job.args) > 1:
            num_array_tasks = len(job.args)
        header = self._create_job_header(
            job.executor, num_array_tasks, job_log_dir, job_name
        )

        return self._create_job_script(job=job, setup=setup, header=header)


class JobClient(abc.ABC):
    _artifact_store: artifacts.ArtifactStore
    _settings: config_lib.ExecutionSettings
    builder_cls: Callable[[config_lib.ExecutionSettings], JobScriptBuilder]

    @abc.abstractmethod
    def _launch(self, job_script_path: str, num_jobs: int) -> Tuple[Optional[str], Any]:
        raise NotImplementedError

    def launch(self, job_name: str, job: ClusterJob):
        job_name = re.sub("\\W", "_", job_name)
        job_script_dir = self._artifact_store.job_path(job_name)
        job_log_dir = self._artifact_store.job_log_path(job_name)
        job_script_builder = self.builder_cls(self._settings)
        job_script_content = job_script_builder.build(job, job_name, job_log_dir)

        self._artifact_store.deploy_job_scripts(job_name, job_script_content)
        job_script_path = os.path.join(job_script_dir, JOB_SCRIPT_NAME)

        if isinstance(job, array_job.ArrayJob):
            num_jobs = len(job.env_vars)
        else:
            num_jobs = 1
        job_id, handles = self._launch(job_script_path, num_jobs)
        if job_id is not None:
            self._save_job_id(job_script_path, job_id)

        return handles

    def _save_job_id(self, job_script_path: str, job_id: str):
        self._artifact_store._fs.write_text(
            os.path.join(os.path.dirname(job_script_path), "job_id"), f"{job_id}\n"
        )
