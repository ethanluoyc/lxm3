import datetime
import os
from typing import List, Optional, Union, cast

import paramiko

from lxm3 import xm
from lxm3.xm_cluster import config as config_lib
from lxm3.xm_cluster import executables
from lxm3.xm_cluster import executors
from lxm3.xm_cluster.execution import artifacts
from lxm3.xm_cluster.execution import job_script


def create_array_job(
    *,
    artifact: artifacts.Artifact,
    executable: executables.Command,
    singularity_image: Optional[str],
    singularity_options: Optional[executors.SingularityOptions],
    jobs: List[xm.Job],
    use_gpu: bool,
    version: Optional[str] = None,
    job_script_shebang: str,
    task_offset: int,
    task_id_var_name: str,
    setup: str,
    header: str,
):
    version = version or datetime.datetime.now().strftime("%Y%m%d.%H%M%S")

    job_name = f"job-{version}"
    array_wrapper = _create_array_wrapper(executable, jobs, task_offset)
    deploy_archive_path = artifact.archive_path(executable.resource_uri)
    setup_cmds = """
set -e

_prepare_workdir() {
    WORKDIR=$(mktemp -t -d -u lxm-workdir.XXXXX)
    echo >&2 "INFO[$(basename $0)]: Prepare work directory: $WORKDIR"
    mkdir -p $WORKDIR

    # Extract archives
    ARCHIVES="%(archives)s"
    for ar in $ARCHIVES; do
        unzip -q -d $WORKDIR $ar
    done

    ARRAY_WRAPPER_PATH=$(mktemp -t -u lxm-array-wrapper.XXXXX)
    echo >&2 "DEBUG[$(basename $0)]: Create wrapper script: $ARRAY_WRAPPER_PATH"
    cat << 'EOF' > $ARRAY_WRAPPER_PATH
%(array_wrapper)s
EOF
    chmod +x $ARRAY_WRAPPER_PATH

    _cleanup() {
        echo >&2 "INFO[$(basename $0)]: Clean up work directory: $WORKDIR"
        rm -rf $WORKDIR
        rm -f $ARRAY_WRAPPER_PATH
    }

    trap _cleanup EXIT

}

_prepare_workdir
cd $WORKDIR
%(setup)s
""" % {
        "array_wrapper": array_wrapper,
        "archives": " ".join([deploy_archive_path]),
        "setup": setup,
    }
    # Always use the array wrapper for but hardcode task id if not launching one job.
    if len(jobs) > 1:
        job_command = " ".join(["sh", "$ARRAY_WRAPPER_PATH", f"${task_id_var_name}"])
    else:
        job_command = " ".join(["sh", "$ARRAY_WRAPPER_PATH", f"{task_offset}"])

    if singularity_image is not None:
        deploy_container_path = artifact.singularity_image_path(
            os.path.basename(singularity_image)
        )
        singularity_options = singularity_options
        job_command = _wrap_singularity_cmd(
            job_command,
            deploy_container_path,
            singularity_options,
            use_gpu,
        )

    job_script_content = job_script.create_job_script(
        job_command,
        header,
        setup=setup_cmds,
        shebang=job_script_shebang,
    )

    return _put_job_resources(
        artifact=artifact,
        executable=executable,
        singularity_image=singularity_image,
        job_name=job_name,
        job_script_content=job_script_content,
    )


def _put_job_resources(
    *,
    artifact: artifacts.Artifact,
    executable: executables.Command,
    singularity_image: Optional[str],
    job_name: str,
    job_script_content: str,
) -> str:
    # Put artifacts on the staging fs
    job_script_dir = artifact.job_path(job_name)
    artifact.deploy_resource_archive(executable.resource_uri)

    if singularity_image is not None:
        artifact.deploy_singularity_container(singularity_image)

    artifact.deploy_job_scripts(job_name, job_script_content)

    deploy_job_script_path = os.path.join(job_script_dir, job_script.JOB_SCRIPT_NAME)

    return deploy_job_script_path


def _create_array_wrapper(
    executable: executables.Command, jobs: List[xm.Job], task_offset: int
):
    work_list = job_script.worklist_from_jobs(executable, jobs)
    return job_script.create_array_wrapper_script(
        cmd=executable.entrypoint_command, work_list=work_list, task_offset=task_offset
    )


def _wrap_singularity_cmd(
    job_command: str,
    deploy_container_path: str,
    singularity_options: Optional[executors.SingularityOptions],
    use_gpu: bool,
) -> str:
    singularity_opts = " ".join(
        job_script.get_singulation_options(singularity_options, use_gpu)
    )
    job_command = (
        f"singularity exec {singularity_opts} {deploy_container_path} {job_command}"
    )
    return job_command


def get_cluster_settings(config: config_lib.Config, jobs: List[xm.Job]):
    executor = cast(Union[executors.Slurm, executors.GridEngine], jobs[0].executor)
    location = executor.requirements.location

    for job in jobs:
        if not isinstance(job.executor, (executors.GridEngine, executors.Slurm)):
            raise ValueError("Only GridEngine and Slurm executors are supported.")
        if job.executor.requirements.location != location:
            raise ValueError("All jobs must be launched on the same cluster.")

    if location is None:
        location = config.default_cluster()

    cluster_config = config.cluster_config(location)
    storage_root = cluster_config["storage"]["staging"]
    hostname = cluster_config.get("server", None)
    user = cluster_config.get("user", None)

    connect_kwargs = {}
    proxycommand = cluster_config.get("proxycommand", None)
    if proxycommand is not None:
        connect_kwargs["sock"] = paramiko.ProxyCommand(proxycommand)

    ssh_private_key = cluster_config.get("ssh_private_key", None)
    if ssh_private_key is not None:
        connect_kwargs["key_filename"] = os.path.expanduser(ssh_private_key)

    password = cluster_config.get("password", None)
    if password is not None:
        connect_kwargs["password"] = password

    return storage_root, hostname, user, connect_kwargs


def write_job_id(artifact, job_script_path: str, job_id: str):
    artifact._fs.write_text(
        os.path.join(os.path.dirname(job_script_path), "job_id"), f"{job_id}\n"
    )
