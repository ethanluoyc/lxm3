import abc
import collections
import shlex
from typing import Dict, Generic, List, Optional, TypedDict, TypeVar, Union, cast

from lxm3 import xm
from lxm3._vendor.xmanager.xm import job_blocks
from lxm3.xm_cluster import array_job
from lxm3.xm_cluster import config as config_lib
from lxm3.xm_cluster import executables
from lxm3.xm_cluster import executors

JOB_SCRIPT_NAME = "job.sh"
CONTAINER_WORKDIR = "/run/task"

JOB_PARAM_NAME = "job-param.sh"
CONTAINER_JOB_PARAM_PATH = f"/etc/{JOB_PARAM_NAME}"

LXM_WORK_DIR = "LXM_WORKDIR"
LXM_TASK_ID = "LXM_TASK_ID"

ClusterJob = Union[xm.Job, array_job.ArrayJob]


class _WorkItem(TypedDict):
    args: List[str]
    env_vars: Dict[str, str]


ExecutorClsType = TypeVar("ExecutorClsType")


def create_singularity_command(
    *,
    image: str,
    args: List[str],
    env_vars: Dict[str, str],
    options: List[str],
    binds: List[str],
    use_gpu: bool,
    pwd: str,
) -> List[str]:
    cmd = ["singularity", "exec"]

    for bind in binds:
        cmd.extend([f"--bind={bind}"])

    for key, value in env_vars.items():
        cmd.extend(["--env", f"{key}={value}"])

    cmd.extend(options)

    if use_gpu:
        cmd.append("--nv")

    cmd.extend(["--pwd", pwd])

    cmd.extend([image, *args])

    return cmd


def create_docker_command(
    image: str,
    args: List[str],
    env_vars: Dict[str, str],
    binds: List[str],
    options: List[str],
    workdir: str,
    use_gpu: bool,
) -> List[str]:
    cmd = ["docker", "run", "--rm"]

    for bind in binds:
        cmd.extend(["--mount", bind])

    cmd.extend(list(options))

    for k, v in env_vars.items():
        cmd.extend(["--env", f"{k}={v}"])

    if use_gpu:
        cmd.extend(
            [
                "--runtime=nvidia",
                "--env",
                '"NVIDIA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"',
            ]
        )

    cmd.extend(["--workdir", workdir])

    cmd.extend([image, *args])

    return cmd


class JobScriptBuilder(abc.ABC, Generic[ExecutorClsType]):
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
        executor: ExecutorClsType,
        num_array_tasks: Optional[int],
        job_log_dir: str,
        job_name: str,
    ) -> str:
        """Create a job header"""

    @classmethod
    @abc.abstractmethod
    def _is_gpu_requested(cls, executor: ExecutorClsType) -> bool:
        """Is GPU requested?"""

    @classmethod
    @abc.abstractmethod
    def _create_setup_cmds(cls, executable, executor: ExecutorClsType) -> str:
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
            lines.append(
                '{key}=$(eval echo \\$"{key}_${lxm_task_id}")'.format(
                    key=key, lxm_task_id=LXM_TASK_ID
                )
            )
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
        lines.append(f'TASK_CMD_ARGS=$(eval echo \\$"TASK_CMD_ARGS_${LXM_TASK_ID}")')
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
                binds = {**executor.singularity_options.bind}
                options = [*executor.singularity_options.extra_options]
            else:
                binds = {}
                options = []

            if self._settings is not None:
                binds.update(
                    self._get_additional_binds(binds, self._settings.singularity.binds)
                )

            binds[f'"${LXM_WORK_DIR}"'] = CONTAINER_WORKDIR
            binds[
                f'"${LXM_WORK_DIR}"/{JOB_PARAM_NAME}'
            ] = f"{CONTAINER_JOB_PARAM_PATH}:ro"
            env_vars = {LXM_TASK_ID: f'"${LXM_TASK_ID}"'}

            entrypoint = create_singularity_command(
                image=executable.singularity_image,
                args=_rewrite_array_job_command(
                    CONTAINER_JOB_PARAM_PATH, executable.entrypoint_command
                ),
                binds=[f"{k}:{v}" for k, v in binds.items()],
                env_vars=env_vars,
                options=options,
                pwd=CONTAINER_WORKDIR,
                use_gpu=self._is_gpu_requested(executor),
            )

        elif executable.docker_image is not None:
            if executor.docker_options is not None:
                binds = {**executor.docker_options.volumes}
                options = [*executor.docker_options.extra_options]
            else:
                binds = {}
                options = []

            mounts = [f"type=bind,source={k},target={v}" for k, v in binds.items()]
            mounts.append(
                f'type=bind,source="${LXM_WORK_DIR}",target={CONTAINER_WORKDIR}'
            )
            mounts.append(
                f'type=bind,source="${LXM_WORK_DIR}"/{JOB_PARAM_NAME},target={CONTAINER_JOB_PARAM_PATH},readonly'
            )

            env_vars = {f"{LXM_TASK_ID}": f'"${LXM_TASK_ID}"'}
            entrypoint = create_docker_command(
                image=executable.docker_image,
                args=_rewrite_array_job_command(
                    CONTAINER_JOB_PARAM_PATH, executable.entrypoint_command
                ),
                env_vars=env_vars,
                binds=mounts,
                options=options,
                workdir=CONTAINER_WORKDIR,
                use_gpu=self._is_gpu_requested(executor),
            )

        else:
            entrypoint = _rewrite_array_job_command(
                f"./{JOB_PARAM_NAME}", executable.entrypoint_command
            )

        cmds = """\
TASK_OFFSET=%(task_offset)s
TASK_INDEX_NAME="%(task_index_name)s"
NUM_TASKS="%(num_tasks)s"

if [ $NUM_TASKS -eq 1 ]; then
    # If there is only one task, then we don't need to use the task index
    %(lxm_task_id)s=0
else
    %(lxm_task_id)s=$(($(eval echo \\$"TASK_INDEX_NAME") - $TASK_OFFSET))
fi
export LXM_TASK_ID
cat <<'EOF' > "$%(lxm_work_dir)s"/%(job_param_name)s
%(array_script)s
EOF
chmod +x "$%(lxm_work_dir)s"/%(job_param_name)s
%(entrypoint)s
""" % {
            "task_offset": self.TASK_OFFSET,
            "task_index_name": self.TASK_ID_VAR_NAME,
            "num_tasks": num_tasks,
            "entrypoint": " ".join(entrypoint),
            "array_script": array_script,
            "job_param_name": JOB_PARAM_NAME,
            "lxm_work_dir": LXM_WORK_DIR,
            "lxm_task_id": LXM_TASK_ID,
        }

        job_script_content = """\
%(shebang)s
%(header)s
%(setup)s
set -e

%(lxm_work_dir)s=$(mktemp -d)
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
  echo >& 2 "DEBUG[$(basename $0)] Cleaning up $%(lxm_work_dir)s"
  rm -rf $%(lxm_work_dir)s
}
trap cleanup EXIT

cd $%(lxm_work_dir)s

%(entrypoint)s
""" % {
            "shebang": self.JOB_SCRIPT_SHEBANG,
            "header": header,
            "setup": setup,
            "entrypoint": cmds,
            "archives": " ".join([executable.resource_uri]),
            "lxm_work_dir": LXM_WORK_DIR,
        }
        return job_script_content
