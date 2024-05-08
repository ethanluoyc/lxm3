import abc
import collections
import os
import shlex
import textwrap
from typing import Dict, Generic, List, Optional, TypeVar, Union, cast

import attr
import fsspec
from fsspec.implementations import sftp

from lxm3 import xm
from lxm3.xm_cluster import array_job
from lxm3.xm_cluster import artifacts
from lxm3.xm_cluster import config as config_lib
from lxm3.xm_cluster import executables
from lxm3.xm_cluster import executors

JobType = Union[xm.Job, array_job.ArrayJob]
ExecutorType = TypeVar("ExecutorType", bound=executors.SupportsContainer)


class JobScriptBuilder(abc.ABC, Generic[ExecutorType]):
    ARRAY_TASK_ID: str
    ARRAY_TASK_OFFSET: int
    JOB_SCRIPT_SHEBANG: str = "#!/usr/bin/env bash"
    JOB_ENV_PATTERN = None

    CONTAINER_WORKDIR: str = "/run/lxm3/workdir"
    JOB_PARAM_NAME: str = "job-param.sh"
    CONTAINER_JOB_PARAM_PATH: str = f"/tmp/{JOB_PARAM_NAME}"

    @classmethod
    @abc.abstractmethod
    def _create_job_script_header(
        cls,
        executor: ExecutorType,
        num_array_tasks: Optional[int],
        job_log_dir: str,
        job_name: str,
    ) -> str:
        """Create a job header"""

    @classmethod
    @abc.abstractmethod
    def _is_gpu_requested(cls, executor: ExecutorType) -> bool:
        """Infer GPU resource from executor"""

    @classmethod
    @abc.abstractmethod
    def _create_job_script_prologue(
        cls,
        executable: executables.AppBundle,
        executor: ExecutorType,
    ) -> str:
        """Generate backend specific setup commands"""

    def _create_job_args_script(self, job: JobType) -> str:
        executable = job.executable
        assert isinstance(executable, executables.AppBundle)
        if isinstance(job, xm.Job):
            args = xm.merge_args(executable.args, job.args).to_list()
            env = {**executable.env_vars, **job.env_vars}
            return _create_job_args_script(args, env)
        elif isinstance(job, array_job.ArrayJob):
            args = [
                xm.merge_args(executable.args, per_task_args).to_list()
                for per_task_args in job.args
            ]
            env = [
                {**executable.env_vars, **per_task_envs}
                for per_task_envs in job.env_vars
            ]

            return _create_array_job_args_script(
                args, env, self.ARRAY_TASK_ID, self.ARRAY_TASK_OFFSET
            )
        else:
            raise ValueError(f"{type(job)} is not supported")

    def _create_install_commands(self, job: JobType, install_dir: str) -> str:
        executable = job.executable
        assert isinstance(executable, executables.AppBundle)
        job_args_script = self._create_job_args_script(job)
        if self.JOB_ENV_PATTERN:
            export_env_file_cmds = (
                f"printenv | {{ grep -E '{self.JOB_ENV_PATTERN}' || :; }} > "
                f'"{install_dir}/.environment"'
            )
        else:
            export_env_file_cmds = f'touch "{install_dir}/.environment"'
        extract_pkg_cmds = _get_extract_command(executable.resource_uri, install_dir)
        save_job_args_cmds = f"cat <<'EOF' > \"{install_dir}/{self.JOB_PARAM_NAME}\"\n{job_args_script}\nEOF"
        return "\n".join(
            [
                extract_pkg_cmds,
                save_job_args_cmds,
                export_env_file_cmds,
            ]
        )

    def _create_entrypoint_commands(self, job: JobType, install_dir: str) -> str:
        executable = job.executable
        if not isinstance(executable, executables.AppBundle):
            raise ValueError("Only Command executable is supported")
        executor = job.executor
        if not isinstance(executor, executors.SupportsContainer):
            raise TypeError("Executor should support container configuration")

        if executable.container_image is not None:
            image = executable.container_image.name
            image_type = executable.container_image.image_type

            if image_type == executables.ContainerImageType.SINGULARITY:
                get_container_cmd = create_singularity_command
                singularity_options = (
                    executor.singularity_options or executors.SingularityOptions()
                )
                bind_mounts = [
                    BindMount(src, dst) for src, dst in singularity_options.bind
                ]
                runtime_options = [*singularity_options.extra_options]
            elif image_type == executables.ContainerImageType.DOCKER:
                get_container_cmd = create_docker_command
                docker_options = executor.docker_options or executors.DockerOptions()
                bind_mounts = [
                    BindMount(src, dst) for src, dst in docker_options.volumes
                ]
                runtime_options = [*docker_options.extra_options]
            else:
                assert False

            bind_mounts.extend(
                [
                    BindMount(install_dir, self.CONTAINER_WORKDIR),
                    BindMount(
                        os.path.join(install_dir, self.JOB_PARAM_NAME),
                        self.CONTAINER_JOB_PARAM_PATH,
                        read_only=True,
                    ),
                ]
            )

            entrypoint = get_container_cmd(
                image=image,
                args=_rewrite_array_job_command(
                    self.CONTAINER_JOB_PARAM_PATH, executable.entrypoint_command
                ),
                bind_mounts=bind_mounts,
                env_vars={},
                options=runtime_options,
                working_dir=self.CONTAINER_WORKDIR,
                use_gpu=self._is_gpu_requested(executor),
                env_file=os.path.join(install_dir, ".environment"),
            )

        else:
            entrypoint = _rewrite_array_job_command(
                f"./{self.JOB_PARAM_NAME}", executable.entrypoint_command
            )

        return " ".join(entrypoint)

    def build(
        self, job: Union[xm.Job, array_job.ArrayJob], job_name: str, job_log_dir: str
    ) -> str:
        executable = job.executable
        if not isinstance(executable, executables.AppBundle):
            raise TypeError("Only AppBundle is supported")
        executor = cast(executors.SupportsContainer, job.executor)

        num_array_tasks = None
        if isinstance(job, array_job.ArrayJob):
            num_array_tasks = len(job.args)

        header = self._create_job_script_header(
            executor, num_array_tasks, job_log_dir, job_name
        )
        prologue = self._create_job_script_prologue(executable, executor)
        install_dir = "$LXM_WORKDIR"
        install_cmds = self._create_install_commands(job, install_dir)
        entrypoint_cmds = self._create_entrypoint_commands(job, install_dir)
        return _JOB_SCRIPT_TEMPLATE % {
            "shebang": self.JOB_SCRIPT_SHEBANG,
            "header": header,
            "install": install_cmds,
            "prologue": prologue,
            "entrypoint": entrypoint_cmds,
        }


_JOB_SCRIPT_TEMPLATE = """\
%(shebang)s
%(header)s
set -e

LXM_WORKDIR="$(mktemp -d)"
cleanup() {
  echo >& 2 "DEBUG[$(basename "$0")] Cleaning up $LXM_WORKDIR"
  rm -rf "$LXM_WORKDIR"
}
trap cleanup EXIT
cd "$LXM_WORKDIR"
%(install)s

%(prologue)s

%(entrypoint)s
"""


@attr.s(auto_attribs=True)
class BindMount:
    path: str
    mount_path: str
    read_only: bool = False


def create_singularity_command(
    *,
    image: str,
    args: List[str],
    env_vars: Dict[str, str],
    options: List[str],
    bind_mounts: List[BindMount],
    use_gpu: bool,
    working_dir: str,
    env_file: str,
) -> List[str]:
    cmd = ["singularity", "exec"]

    for mount in bind_mounts:
        bm = f"{mount.path}:{mount.mount_path}"
        if mount.read_only:
            bm += ":ro"
        cmd.append(f'--bind="{bm}"')

    for key, value in env_vars.items():
        cmd.append(f'--env={key}="{value}"')

    cmd.extend(options)

    if use_gpu:
        cmd.append("--nv")

    cmd.append(f'--pwd="{working_dir}"')
    cmd.append(f'--env-file="{env_file}"')

    cmd.extend([image, *args])

    return cmd


def create_docker_command(
    *,
    image: str,
    args: List[str],
    env_vars: Dict[str, str],
    bind_mounts: List[BindMount],
    options: List[str],
    working_dir: str,
    use_gpu: bool,
    env_file: str,
) -> List[str]:
    cmd = ["docker", "run", "--rm"]

    for mount in bind_mounts:
        mount_spec = f"type=bind,source={mount.path},target={mount.mount_path}"
        if mount.read_only:
            mount_spec += ",readonly"
        cmd.extend([f'--mount="{mount_spec}"'])

    for k, v in env_vars.items():
        cmd.append(f'--env={k}="{v}"')

    if use_gpu:
        cmd.extend(
            [
                "--runtime=nvidia",
                '--env=NVIDIA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-all}"',
            ]
        )

    cmd.extend(options)
    cmd.append(f'--workdir="{working_dir}"')
    cmd.append(f'--env-file="{env_file}"')
    cmd.extend([image, *args])

    return cmd


def _rewrite_array_job_command(array_script_path: str, cmd: str) -> List[str]:
    return ["sh", "-c", shlex.quote(f'. {array_script_path}; {cmd} "$@"')]


def _get_extract_command(archive: str, directory: str) -> str:
    if archive.endswith(".zip"):
        return f'unzip -q -d "{directory}" "{archive}"'
    elif archive.endswith(".tar"):
        return f'tar -C "{directory}" -xf "{archive}"'
    elif archive.endswith((".tar.gz", ".tgz")):
        return f'tar -C "{directory}" -xzf "{archive}"'
    else:
        raise ValueError(archive)


def _create_job_args_script(args: List[str], env: Dict[str, str]) -> str:
    return "\n".join(
        [
            "export LXM_TASK_ID=0",
            _create_env_vars([env], "LXM_TASK_ID", 0),
            _create_args([args], "LXM_TASK_ID", 0),
        ]
    )


def _create_array_job_args_script(
    args: List[List[str]],
    env: List[Dict[str, str]],
    index_name: str,
    index_offset: int,
) -> str:
    return "\n".join(
        [
            textwrap.dedent(
                f"""\
                if [ -z ${{{index_name}+x}} ];
                then
                echo >&2 "ERROR[$0]: \\${index_name} is not set."
                exit 2
                fi"""
            ),
            _create_env_vars(env, index_name, index_offset),
            _create_args(args, index_name, index_offset),
        ]
    )


def _create_env_vars(
    env_vars_list: List[Dict[str, str]], index_name: str, index_offset: int
) -> str:
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
            raise ValueError("Expect all environment variables to have the same keys")

    # Generate shared environment variables
    for k in sorted(common_keys):
        lines.append(f'export {k}="{env_vars_list[0][k]}"')

    for key in first_keys:
        if key in common_keys:
            continue
        for task_id, env_vars in enumerate(env_vars_list, start=index_offset):
            lines.append(f'{key}_{task_id}="{env_vars[key]}"')
        lines.append(f'{key}=$(eval echo \\$"{key}_${index_name}")')
        lines.append(f"export {key}")
    content = "\n".join(lines)
    return content


def _create_args(args_list: List[List[str]], index_name: str, index_offset: int) -> str:
    """Create the args list."""
    if not args_list:
        return ""
    lines = []
    for task_id, args in enumerate(args_list, start=index_offset):
        args_str = " ".join([a for a in args])
        lines.append(f'TASK_CMD_ARGS_{task_id}="{args_str}"')
    lines.append(f'TASK_CMD_ARGS=$(eval echo \\$"TASK_CMD_ARGS_${index_name}")')
    lines.append("eval set -- $TASK_CMD_ARGS")
    content = "\n".join(lines)
    return content


def job_path(job_name: str):
    return os.path.join("jobs", job_name)


def job_script_path(job_name: str):
    return os.path.join(job_path(job_name), "job.sh")


def job_log_path(job_name: str):
    return os.path.join("logs", job_name)


def flatten_job(
    job: Union[xm.JobGroup, array_job.ArrayJob],
) -> List[Union[xm.Job, array_job.ArrayJob]]:
    if isinstance(job, array_job.ArrayJob):
        return [job]  # type: ignore
    elif isinstance(job, xm.JobGroup):
        jobs = xm.job_operators.flatten_jobs(job)
        if len(jobs) > 1:
            raise NotImplementedError("JobGroup is not supported.")
        return jobs  # type: ignore
    else:
        raise NotImplementedError()


def create_artifact_store(project, settings):
    project = config_lib.default().project()
    settings = config_lib.default().cluster_settings()
    hostname = settings.hostname
    storage_root = settings.storage_root
    user = settings.user
    ssh_config = settings.ssh_config

    if hostname is None:
        filesystem = fsspec.filesystem("file")
        storage_root = os.path.abspath(os.path.expanduser(storage_root))
    else:
        filesystem = sftp.SFTPFileSystem(host=hostname, username=user, **ssh_config)
        storage_root = filesystem.ftp.normalize(storage_root)

    return artifacts.ArtifactStore(
        filesystem, staging_directory=storage_root, project=project
    )
