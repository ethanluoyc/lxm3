import datetime
import os
import shlex
import shutil
from typing import List

import termcolor
from absl import logging

from lxm3 import xm
from lxm3._vendor.xmanager.xm import job_blocks
from lxm3.clusters import gridengine
from lxm3.xm_cluster import executables
from lxm3.xm_cluster import executors

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


def _create_env_vars(env_vars_list):
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


def _create_args(args_list):
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


def _get_cmd_str(cmd):
    if isinstance(cmd, (list, tuple)):
        return " ".join(list(map(shlex.quote, cmd)))
    return cmd


def create_array_wrapper_script(
    cmd,
    work_list,
    task_offset,
    shebang="#!/usr/bin/env bash",
):
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


def deploy_resource_archive(fs, storage_root, executable):
    archive_dir = os.path.join(storage_root, "archives")
    archive_name = os.path.basename(executable.resource_uri)
    deploy_archive_path = os.path.join(archive_dir, archive_name)
    if not fs.exists(deploy_archive_path):
        fs.makedirs(archive_dir, exist_ok=True)
        print(termcolor.colored(f"Deploying archive to {deploy_archive_path}", "cyan"))
        fs.put_file(executable.resource_uri, deploy_archive_path)
    return deploy_archive_path


def _create_file(fs, path, content):
    with fs.open(path, "wt") as f:
        f.write(content)


def _create_job_script(cmd, header, archives=None, setup=""):
    if archives is None:
        archives = []
    return """\
#!/usr/bin/env bash
%(header)s

set -e

_prepare_workdir() {
    WORKDIR=$(mktemp -t -d -u lxm-workdir.XXXXX)
    echo "Prepare work directory: $WORKDIR"
    mkdir -p $WORKDIR

    _cleanup() {
        echo "Clean up work directory: $WORKDIR"
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


def deploy_singularity_container(fs, storage_root, singularity_image):
    from fsspec.implementations.local import LocalFileSystem

    if isinstance(fs, LocalFileSystem):
        # When running on local file system, we can just use the image path
        logging.info("Running on local file system, using image path")
        return os.path.abspath(singularity_image)

    container_dir = os.path.join(storage_root, "containers")
    if not fs.exists(container_dir):
        fs.makedirs(container_dir, exist_ok=True)
    image_name = os.path.basename(singularity_image)
    deploy_container_path = os.path.join(container_dir, image_name)
    if not fs.exists(deploy_container_path):
        print(termcolor.colored(f"Uploading container {image_name}...", "cyan"))
        fs.put(singularity_image, deploy_container_path)
        print(
            termcolor.colored(
                f"  Singularity container available at {deploy_container_path}",
                "cyan",
            )
        )
    else:
        if fs.modified(deploy_container_path) < datetime.datetime.fromtimestamp(
            os.path.getmtime(singularity_image)
        ):
            print(
                termcolor.colored(
                    f"Local container is newer. Uploading container {image_name}...",
                    "cyan",
                )
            )
            fs.put(singularity_image, deploy_container_path)
            print(
                termcolor.colored(
                    f"  Singularity container available at {deploy_container_path}",
                    "cyan",
                )
            )
        else:
            print(
                termcolor.colored(
                    f"  Container {image_name} exists, skipping upload.", "blue"
                )
            )
    return deploy_container_path


def _generate_header_from_executor(
    job_name: str,
    executor: executors.GridEngine,
    num_array_tasks: int,
    job_script_dir: str,
):
    header = []

    header.append(f"#$ -N {job_name}")

    for resource, value in executor.requirements.resources.items():
        header.append(f"#$ -l {resource}={value}")

    for pe_name, value in executor.parallel_environments.items():
        header.append(f"#$ -pe {pe_name} {value}")

    if executor.walltime:
        if not isinstance(executor.walltime, str):
            duration = datetime.timedelta(seconds=executor.walltime)
        else:
            duration = executor.walltime
        header.append(f"#$ -l h_rt={duration}")

    if executor.queue:
        header.append(f"#$ -q {executor.queue}")

    reserved = executor.reserved
    if reserved is None and executor.parallel_environments:
        reserved = True

    if reserved:
        header.append("#$ -R")

    log_directory = executor.log_directory or os.path.join(job_script_dir, "logs")
    stdout = os.path.join(log_directory, "$JOB_NAME.o$JOB_ID.$TASK_ID")
    header.append(f"#$ -o {stdout}")
    stderr = os.path.join(log_directory, "$JOB_NAME.e$JOB_ID.$TASK_ID")
    header.append(f"#$ -e {stderr}")
    if executor.merge_output:
        header.append("#$ -j y")

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


def deploy_job_resources(fs, storage_root, jobs):
    executable = jobs[0].executable
    executor = jobs[0].executor

    assert isinstance(executor, executors.GridEngine) or isinstance(
        executor, executors.Local
    )
    assert isinstance(executable, executables.Command)

    for job in jobs[1:]:
        if job.executable != executable:
            raise ValueError("All jobs must have the same executable.")
        if job.executor != executor:
            raise ValueError("All jobs must have the same executor.")

    singularity_image = executor.singularity_container
    singularity_opts = " ".join(executor.singularity_options)
    if isinstance(executor, executors.GridEngine):
        if (
            "gpu" in executor.parallel_environments
            or "gpu" in executor.requirements.resources
        ):
            singularity_opts += " --nv"
    elif isinstance(executor, executors.Local):
        if shutil.which("nvidia-smi"):
            singularity_opts += " --nv"
    else:
        pass

    deploy_archive_path = deploy_resource_archive(fs, storage_root, executable)

    version = datetime.datetime.now().strftime("%Y%m%d.%H%M%S")
    job_script_dir = os.path.join(storage_root, f"job_scripts/job-{version}")

    fs.makedirs(job_script_dir, exist_ok=True)

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

    array_wrapper = create_array_wrapper_script(
        cmd=executable.entrypoint_command, work_list=work_list, task_offset=1
    )

    array_wrapper_path = os.path.join(job_script_dir, "array_wrapper.sh")
    _create_file(fs, array_wrapper_path, array_wrapper)
    setup_cmds = ["hostname"]
    job_command = " ".join(["sh", os.fspath(array_wrapper_path), "$SGE_TASK_ID"])

    if singularity_image is not None:
        deploy_container_path = deploy_singularity_container(
            fs, storage_root, singularity_image
        )
        job_command = (
            f"singularity exec {singularity_opts} {deploy_container_path} {job_command}"
        )
        setup_cmds.append("singularity --version")

    job_script_path = os.path.join(job_script_dir, "job.sh")
    fs.makedirs(os.path.join(job_script_dir, "logs"), exist_ok=True)

    if isinstance(executor, executors.GridEngine):
        job_name = jobs[0].name

        job_header = _generate_header_from_executor(
            job_name, executor, len(work_list), job_script_dir
        )

    elif isinstance(executor, executors.Local):
        job_header = ""

    job_script = _create_job_script(
        job_command,
        job_header,
        archives=[deploy_archive_path],
        setup="\n".join(setup_cmds),
    )
    _create_file(fs, job_script_path, job_script)
    print(termcolor.colored(f"Created job script at {job_script_path}", "cyan"))
    return job_script_path


class GridEngineHandle:
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id

    async def wait(self):
        raise NotImplementedError()

    async def monitor(self):
        raise NotImplementedError()


async def launch(hostname, user, fs, storage_root, jobs: List[xm.Job]):
    if len(jobs) < 1:
        return []

    print(termcolor.colored(f"Launching {len(jobs)} jobs", "cyan"))
    job_script_path = deploy_job_resources(fs, storage_root, jobs)

    print(termcolor.colored(f"Launch with command:\n  qsub {job_script_path}", "cyan"))
    client = gridengine.Client(hostname, user)
    job_ids = client.launch(job_script_path)
    return [GridEngineHandle(job_id) for job_id in job_ids]
