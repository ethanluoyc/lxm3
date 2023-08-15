import abc
import datetime
import os
import subprocess
from typing import List, Optional, Sequence

import fsspec
import rich.progress
import rich.syntax

from lxm3.xm_cluster.console import console


def rsync(
    src: str,
    dst: str,
    opt: List[str],
    host: Optional[str] = None,
    excludes: Optional[Sequence[str]] = None,
    filters: Optional[Sequence[str]] = None,
    mkdirs: bool = False,
):
    if excludes is None:
        excludes = []
    if filters is None:
        filters = []
    opt = list(opt)
    for exclude in excludes:
        opt.append(f"--exclude={exclude}")
    for filter in filters:
        opt.append(f"--filter=:- {filter}")
    if not host:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        sync_cmd = ["rsync"] + opt + [src, dst]
        console.log("Running", " ".join(sync_cmd))
        subprocess.check_call(sync_cmd)
    else:
        if mkdirs:
            subprocess.check_output(["ssh", host, "mkdir", "-p", os.path.dirname(dst)])
        dst = f"{host}:{dst}"
        sync_cmd = ["rsync"] + opt + [src, dst]
        console.log(
            rich.syntax.Syntax(f"rsync {' '.join(opt)} \\\n  {src} \\\n  {dst}", "bash")
        )
        subprocess.check_call(sync_cmd)


class Artifact(abc.ABC):
    def __init__(self, filesystem, staging_directory: str, project=None):
        if project:
            self._storage_root = os.path.join(staging_directory, "projects", project)
        else:
            self._storage_root = staging_directory
        self._fs = filesystem

    def job_path(self, job_name: str):
        return os.path.join(self._storage_root, "jobs", job_name)

    def job_script_path(self, job_name: str):
        return os.path.join(self.job_path(job_name), "job.sh")

    def job_array_wrapper_path(self, job_name: str):
        return os.path.join(self.job_path(job_name), "array_wrapper.sh")

    def singularity_image_path(self, image_name: str):
        return os.path.join(self._storage_root, "containers", image_name)

    def archive_path(self, resource_uri: str):
        return os.path.join(
            self._storage_root, "archives", os.path.basename(resource_uri)
        )

    def _should_update(self, src: str, dst: str) -> bool:
        if not self._fs.exists(dst):
            return True

        local_stat = os.stat(src)
        local_mtime = datetime.datetime.utcfromtimestamp(
            local_stat.st_mtime
        ).timestamp()
        storage_stat = self._fs.info(dst)
        storage_mtime = storage_stat["mtime"]

        if isinstance(storage_mtime, datetime.datetime):
            storage_mtime = storage_mtime.timestamp()

        if local_stat.st_size != storage_stat["size"]:
            return True

        if int(local_mtime) > int(storage_mtime):
            return True

        return False

    def _put_content(self, dst, content):
        self._fs.makedirs(os.path.dirname(dst), exist_ok=True)

        with self._fs.open(dst, "wt") as f:
            f.write(content)

    def _put_file(self, local_filename, dst):
        self._fs.makedirs(os.path.dirname(dst), exist_ok=True)
        self._fs.put(local_filename, dst)

    def deploy_job_scripts(self, job_name, job_script, array_wrapper):
        job_path = self.job_path(job_name)
        job_log_path = os.path.join(job_path, "logs")

        self._fs.makedirs(job_path, exist_ok=True)
        self._fs.makedirs(job_log_path, exist_ok=True)
        self._put_content(self.job_script_path(job_name), job_script)
        self._put_content(self.job_array_wrapper_path(job_name), array_wrapper)
        console.log(f"Created job script {self.job_script_path(job_name)}")

    def deploy_singularity_container(self, singularity_image):
        image_name = os.path.basename(singularity_image)
        deploy_container_path = self.singularity_image_path(image_name)
        should_update = self._should_update(singularity_image, deploy_container_path)
        if should_update:
            self._fs.makedirs(os.path.dirname(deploy_container_path), exist_ok=True)
            self._put_file(singularity_image, deploy_container_path)
            console.log(f"Deployed Singularity container to {deploy_container_path}")
        else:
            console.log(f"Container {deploy_container_path} exists, skip upload.")
        return deploy_container_path

    def deploy_resource_archive(self, resource_uri):
        deploy_archive_path = self.archive_path(resource_uri)

        should_update = self._should_update(resource_uri, deploy_archive_path)
        if should_update:
            self._put_file(resource_uri, deploy_archive_path)
            console.log(f"Deployed archive to {deploy_archive_path}")
        else:
            console.log(f"Archive {deploy_archive_path} exists, skipping upload.")
        return deploy_archive_path


class LocalArtifact(Artifact):
    def __init__(self, staging_directory: str, project=None):
        if not os.path.exists(staging_directory):
            os.makedirs(staging_directory)
            # Create a .gitignore file to prevent git from tracking the directory
            with open(os.path.join(staging_directory, ".gitignore"), "wt") as f:
                f.write("*\n")
        super().__init__(fsspec.filesystem("file"), staging_directory, project)


class RemoteArtifact(Artifact):
    def __init__(self, hostname, user, staging_directory: str, project=None):
        fs = fsspec.filesystem("sftp", host=hostname, username=user)
        # Normalize the storage root to an absolute path.
        self._host = hostname
        self._user = user
        if not os.path.isabs(staging_directory):
            staging_directory = fs.ftp.normalize(staging_directory)
        super().__init__(fs, staging_directory, project)

    def deploy_singularity_container(self, singularity_image):
        image_name = os.path.basename(singularity_image)
        deploy_container_path = self.singularity_image_path(image_name)
        should_update = self._should_update(singularity_image, deploy_container_path)
        if should_update:
            with rich.progress.Progress(
                rich.progress.TextColumn("[progress.description]{task.description}"),
                rich.progress.BarColumn(),
                rich.progress.TaskProgressColumn(),
                rich.progress.TimeRemainingColumn(),
                rich.progress.TransferSpeedColumn(),
                console=console,
            ) as progress:
                self._fs.makedirs(os.path.dirname(deploy_container_path), exist_ok=True)

                task = progress.add_task(
                    f"Uploading {os.path.basename(singularity_image)}"
                )

                def callback(transferred_bytes: int, total_bytes: int):
                    progress.update(
                        task, completed=transferred_bytes, total=total_bytes
                    )

                self._fs.ftp.put(
                    singularity_image,
                    deploy_container_path,
                    callback=callback,
                    confirm=True,
                )
                progress.update(task, description="Done!")

        else:
            console.log(f"Container {deploy_container_path} exists, skip upload.")
        return deploy_container_path
