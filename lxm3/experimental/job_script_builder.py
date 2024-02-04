import collections
import shlex
from typing import Any, Dict, List, Optional, Union


class JobScriptBuilder:
    ARRAY_TASK_VAR_NAME = "SGE_TASK_ID"
    ARRAY_TASK_OFFSET = 1
    DIRECTIVE_PREFIX = "#$"
    JOB_ENV_PATTERN = "^(JOB_|SGE_|PE|NSLOTS|NHOSTS)"

    @classmethod
    def _create_array_task_env_vars(cls, env_vars_list: List[Dict[str, str]]) -> str:
        """Create the env_vars list."""
        lines = []
        # TODO(yl): Handle the case where each task has different env_vars
        first_keys = set(env_vars_list[0].keys())
        if not first_keys:
            return ""
        for env_vars in env_vars_list:
            if first_keys != set(env_vars.keys()):
                raise ValueError(
                    "Expect all task environment variables to have the same keys"
                    "If you want to have different environment variables for each task, set them to ''"
                )

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

        # Generate shared environment variables
        for k in sorted(common_keys):
            lines.append(
                'export {key}="{value}"'.format(key=k, value=env_vars_list[0][k])
            )

        for key in first_keys:
            if key in common_keys:
                continue

        for key in first_keys:
            for task_id, env_vars in enumerate(env_vars_list):
                lines.append(
                    "{key}_{task_id}={value}".format(
                        key=key,
                        task_id=task_id + cls.ARRAY_TASK_OFFSET,
                        value=env_vars[key],
                    )
                )
            lines.append(
                '{key}=$(eval echo \\$"{key}_${task_id_env_var}")'.format(
                    key=key, task_id_env_var=cls.ARRAY_TASK_VAR_NAME
                )
            )
            lines.append("export {key}".format(key=key))
        content = "\n".join(lines)
        return content

    @classmethod
    def _create_array_task_args(cls, args_list: List[List[str]]) -> str:
        """Create the args list."""
        if not args_list:
            return "\n".join(["TASK_CMD_ARGS=''"])
        lines = []
        for task_id, args in enumerate(args_list):
            args_str = shlex.quote(" ".join(args))
            lines.append(
                "TASK_CMD_ARGS_{task_id}={args_str}".format(
                    task_id=task_id + cls.ARRAY_TASK_OFFSET, args_str=args_str
                )
            )
        lines.append(
            'TASK_CMD_ARGS=$(eval echo \\$"TASK_CMD_ARGS_${task_id_env_var}")'.format(
                task_id_env_var=cls.ARRAY_TASK_VAR_NAME
            )
        )
        content = "\n".join(lines)
        return content

    @classmethod
    def create_array_task_script(
        cls,
        per_task_args: List[List[str]],
        per_task_envs: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Create a wrapper script to run an array of tasks.
        Args:
            per_task_args: A list of list[str] where each list contains the command line
                arguments for a task. The arguments will NOT be quoted. If you need to evaluate
                the arguments, use shlex if you need to quote individual arguments.
            per_task_env: A list of dictionaries where each dictionary contains the
                environment variables for a task. If None, then the environment variables
                will be empty.
        Raises:
            ValueError: If the number of environment variables and arguments do not match.
            ValueError: If the environment variables do not shared the same keys.

        Returns:
            A string that represents the wrapper script. This script can be evaluated
            via sh -c. Note that the a shebang is not included.
        """
        if per_task_envs is None:
            per_task_envs = [{} for _ in per_task_args]

        if len(per_task_envs) != len(per_task_args):
            raise ValueError("Expect the same number of env_vars and args")

        env_vars = cls._create_array_task_env_vars(per_task_envs)
        args = cls._create_array_task_args(per_task_args)

        return f"""\
if [ -z ${{{cls.ARRAY_TASK_VAR_NAME}+x}} ];
then
echo >&2 "ERROR[$0]: \\${cls.ARRAY_TASK_VAR_NAME} is not set."
exit 2
fi

# Prepare environment variables for task
{env_vars}

# Prepare command line arguments for task
{args}

# Evaluate the TASK_CMD_ARGS and set the positional arguments
# This is necessary to handle quoting in the arguments.
eval set -- $TASK_CMD_ARGS
"""

    @classmethod
    def create_job_script(
        cls,
        *,
        command: List[str],
        singularity_image: Optional[str] = None,
        singularity_binds: Optional[List[str]] = None,
        singularity_options: Optional[List[str]] = None,
        archives: Optional[Union[str, List[str]]] = None,
        files: Optional[List[str]] = None,
        per_task_args: Optional[List[List[str]]] = None,
        per_task_envs: Optional[List[Dict[str, str]]] = None,
        # SGE options
        resources: Optional[Dict[str, Any]] = None,
        parallel_environments: Optional[Dict[str, int]] = None,
        log_directory: Optional[str] = None,
        merge_log_output: bool = True,
        task_concurrency: Optional[int] = None,
        extra_directives: Optional[List[str]] = None,
        # Additional script content
        prologue: str = "",
        epilogue: str = "",
        verbose: bool = False,
    ):
        shebang = "#!/bin/sh"
        script = [shebang]

        if verbose:
            script.append("LXM_DEBUG=1")

        if not resources:
            resources = {}

        for resource_name, resource_value in resources.items():
            script.append(f"{cls.DIRECTIVE_PREFIX} -l {resource_name}={resource_value}")

        if parallel_environments is None:
            parallel_environments = {}

        for pe_name, pe_value in parallel_environments.items():
            script.append(f"{cls.DIRECTIVE_PREFIX} -pe {pe_name}={pe_value}")

        if log_directory is not None:
            script.append(f"{cls.DIRECTIVE_PREFIX} -o {log_directory}")
            script.append(f"{cls.DIRECTIVE_PREFIX} -e {log_directory}")

        if merge_log_output:
            script.append(f"{cls.DIRECTIVE_PREFIX} -j y")

        if task_concurrency is not None:
            if not task_concurrency > 0:
                raise ValueError("Task concurrency must be positive")

            script.append(f"{cls.DIRECTIVE_PREFIX} -tc {task_concurrency}")

        if per_task_args is not None:
            script.append(f"{cls.DIRECTIVE_PREFIX} -t 1-{len(per_task_args)}")

        if not extra_directives:
            extra_directives = []
        for h in extra_directives:
            script.append(f"{cls.DIRECTIVE_PREFIX} {h}")

        script.append(
            """\
LXM_WORKDIR="$(mktemp -d)"
if [ "${LXM_DEBUG:-}" = 1 ]; then
    echo >& 2 "DEBUG[$(basename "$0")] Working directory: $LXM_WORKDIR"
fi
cleanup() {
    if [ "${LXM_DEBUG:-}" = 1 ]; then
      echo >& 2 "DEBUG[$(basename "$0")] Cleaning up $LXM_WORKDIR"
    fi
    rm -rf "$LXM_WORKDIR"
}
trap cleanup EXIT
cd "$LXM_WORKDIR"
"""
        )

        script.append(prologue)
        if archives:
            if isinstance(archives, str):
                archives = [archives]
            script.append(_extract_archive_cmds(archives, "$LXM_WORKDIR"))
        if files:
            for f in files:
                script.append(f'cp {f} "$LXM_WORKDIR"/')

        job_script_file = "job-params.sh"
        if per_task_args is not None:
            script.append(
                f"""\
cat <<'EOF' > "$LXM_WORKDIR/{job_script_file}"
{cls.create_array_task_script(per_task_args, per_task_envs)}
EOF
chmod +x "$LXM_WORKDIR"/{job_script_file}
"""
            )

        if singularity_image is not None:
            workdir_mount_path = "/run/task"
            container_job_script_path = f"/etc/{job_script_file}"
            env_file = ".env"
            if singularity_binds is None:
                singularity_binds = []
            binds = [
                *singularity_binds,
                f'"$LXM_WORKDIR":{workdir_mount_path}',
            ]
            script.append(
                f"""\
# Save and pass environment variables from SGE to a file
printenv | grep -s -E "{cls.JOB_ENV_PATTERN}" > {env_file}
"""
            )
            if per_task_args is not None:
                binds.append(
                    f'"$LXM_WORKDIR"/{job_script_file}:{container_job_script_path}:ro',
                )
                wrapped_cmd = [
                    "sh",
                    "-c",
                    shlex.quote(
                        f'. {container_job_script_path}; {shlex.join(command)} "$@"'
                    ),
                ]
            else:
                wrapped_cmd = ["sh", "-c", shlex.quote(f'{shlex.join(command)} "$@"')]
            script.append(
                _get_singularity_command(
                    singularity_image,
                    command=wrapped_cmd,
                    binds=binds,
                    env_file=env_file,
                    options=singularity_options,
                    pwd=workdir_mount_path,
                )
            )
        else:
            if per_task_args is not None:
                script.append(
                    f'. "$LXM_WORKDIR"/{job_script_file}; {shlex.join(command)} "$@"'
                )
            else:
                script.append(shlex.join(command))

        script.append(epilogue)
        return "\n".join(script)


create_job_script = JobScriptBuilder.create_job_script


def _get_singularity_command(
    image: str,
    *,
    command: List[str],
    binds: Optional[List[str]] = None,
    envs: Optional[Dict[str, str]] = None,
    options: Optional[List[str]] = None,
    env_file: Optional[str] = None,
    pwd: Optional[str] = None,
):
    if not binds:
        binds = []

    if not options:
        options = []

    if not envs:
        envs = {}

    singularity_cmd = []
    singularity_cmd = [
        "singularity",
        "exec",
        *([f"--env-file={env_file}"] if env_file else []),
        *[f"--env={key}={value}" for key, value in envs.items()],
        *([f"--pwd={pwd}"] if pwd else []),
        *options,
        *[f"--bind={bind}" for bind in binds],
        image,
        *command,
    ]
    return " ".join(singularity_cmd)


def _extract_archive_cmds(archive: List[str], extract_dir: str):
    return """\
# Extract archives
extract_archive() {{
    case $ar in
    *.zip)
        unzip -q -u -d "$1" "$2"
        ;;
    *.tar)
        tar -C "$1" -xf "$2"
        ;;
    *.tar.gz|*.tgz)
        tar -C "$1" -xzf "$2"
        ;;
    *)
        echo >& 2 "Unsupported archive format: $2"
        exit 2
        ;;
    esac
}}
ARCHIVES="{archive}"
for ar in $ARCHIVES; do
    extract_archive "{extract_dir}" "$ar"
done
        """.format(
        archive=" ".join(archive),
        extract_dir=extract_dir,
    )
