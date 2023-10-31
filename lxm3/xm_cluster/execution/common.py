import copy
import os
from typing import Dict, List, Optional, Union

from lxm3 import xm
from lxm3.xm_cluster import config as config_lib
from lxm3.xm_cluster import executables
from lxm3.xm_cluster import executors
from lxm3.xm_cluster.execution import job_script


def _apply_env_overrides(job: xm.Job, env_overrides: Dict[str, str]):
    job.env_vars = copy.copy(job.env_vars)
    for k, v in env_overrides.items():
        if k not in job.env_vars:
            job.env_vars[k] = v


def create_array_job(
    *,
    executable: executables.Command,
    singularity_image: Optional[str],
    singularity_options: Optional[executors.SingularityOptions],
    jobs: List[xm.Job],
    use_gpu: bool,
    job_script_shebang: str,
    task_offset: int,
    task_id_var_name: str,
    setup: str,
    header: str,
    settings: Optional[
        Union[config_lib.LocalSettings, config_lib.ClusterSettings]
    ] = None,
):
    if settings is not None:
        # Apply cluster env overrides
        for job in jobs:
            _apply_env_overrides(job, settings.env)

    if singularity_image is not None and settings is not None:
        for job in jobs:
            _apply_env_overrides(job, settings.singularity.env)

        if singularity_options is None:
            singularity_options = executors.SingularityOptions()
        else:
            singularity_options = copy.deepcopy(singularity_options)
        for src, dst in settings.singularity.binds.items():
            singularity_options.bind.update({src: dst})

    array_wrapper = _create_array_wrapper(executable, jobs, task_offset)
    deploy_archive_path = executable.resource_uri
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
        job_command = _wrap_singularity_cmd(
            job_command,
            singularity_image,
            singularity_options,
            use_gpu,
        )

    job_script_content = job_script.create_job_script(
        job_command,
        header,
        setup=setup_cmds,
        shebang=job_script_shebang,
    )
    return job_script_content


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


def write_job_id(artifact, job_script_path: str, job_id: str):
    artifact._fs.write_text(
        os.path.join(os.path.dirname(job_script_path), "job_id"), f"{job_id}\n"
    )
