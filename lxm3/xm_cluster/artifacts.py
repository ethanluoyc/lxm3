import abc
import datetime
import logging
import os
from typing import Any, Mapping, Optional

import fsspec
import rich.progress
import rich.syntax
from fsspec.implementations.sftp import SFTPFileSystem

from lxm3.xm_cluster.console import console

# Disable verbose logging from paramiko
logging.getLogger("paramiko").setLevel(logging.WARNING)


class ArtifactStore(abc.ABC):
    def __init__(
        self,
        filesystem: fsspec.AbstractFileSystem,
        staging_directory: str,
        project: Optional[str] = None,
    ):
        if project:
            self._storage_root = os.path.join(staging_directory, "projects", project)
        else:
            self._storage_root = staging_directory
        self._fs = filesystem
        self._initialize()

    def _initialize(self):
        pass

    def job_path(self, job_name: str):
        return os.path.join(self._storage_root, "jobs", job_name)

    def job_script_path(self, job_name: str):
        return os.path.join(self.job_path(job_name), "job.sh")

    def job_log_path(self, job_name: str):
        return os.path.join(os.path.join(self._storage_root, "logs", job_name))

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

    def deploy_job_scripts(self, job_name, job_script):
        job_path = self.job_path(job_name)
        job_log_path = self.job_log_path(job_name)

        self._fs.makedirs(job_path, exist_ok=True)
        self._fs.makedirs(job_log_path, exist_ok=True)
        self._put_content(self.job_script_path(job_name), job_script)
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


class LocalArtifactStore(ArtifactStore):
    def __init__(self, staging_directory: str, project=None):
        staging_directory = os.path.abspath(os.path.expanduser(staging_directory))
        super().__init__(fsspec.filesystem("file"), staging_directory, project)

    def _initialize(self):
        if not os.path.exists(self._storage_root):
            self._fs.makedirs(self._storage_root)
            # Create a .gitignore file to prevent git from tracking the directory
            self._fs.write_text(os.path.join(self._storage_root, ".gitignore"), "*\n")

    def deploy_singularity_container(self, singularity_image):
        image_name = os.path.basename(singularity_image)
        deploy_container_path = self.singularity_image_path(image_name)
        should_update = self._should_update(singularity_image, deploy_container_path)
        if should_update:
            self._fs.makedirs(os.path.dirname(deploy_container_path), exist_ok=True)
            if os.path.exists(deploy_container_path):
                os.unlink(deploy_container_path)
            self._put_file(singularity_image, deploy_container_path)
            console.log(f"Deployed Singularity container to {deploy_container_path}")
        else:
            console.log(f"Container {deploy_container_path} exists, skip upload.")
        return deploy_container_path


class RemoteArtifactStore(ArtifactStore):
    _fs: SFTPFileSystem

    def __init__(
        self,
        hostname: str,
        user: Optional[str],
        staging_directory: str,
        project: Optional[str] = None,
        connect_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        if connect_kwargs is None:
            connect_kwargs = {}
        fs = fsspec.filesystem("sftp", host=hostname, username=user, **connect_kwargs)
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


def create_artifact_store(
    storage_root: str,
    hostname: Optional[str] = None,
    user: Optional[str] = None,
    project: Optional[str] = None,
    connect_kwargs=None,
) -> ArtifactStore:
    if hostname is None:
        return LocalArtifactStore(storage_root, project=project)
    else:
        return RemoteArtifactStore(
            hostname, user, storage_root, project=project, connect_kwargs=connect_kwargs
        )
