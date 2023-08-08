import abc
import os

import subprocess
import datetime
import fsspec
from typing import List, Optional, Sequence

from lxm3.xm_cluster.console import console
import rich.syntax


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
    def __init__(self, staging_directory: str, project=None):
        if project:
            self._storage_root = os.path.join(staging_directory, "projects", project)
        else:
            self._storage_root = staging_directory

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

    @abc.abstractmethod
    def deploy_job_scripts(self, job_name, job_script, array_wrapper):
        raise NotImplementedError()

    @abc.abstractmethod
    def deploy_singularity_container(self, singularity_image):
        raise NotImplementedError()

    @abc.abstractmethod
    def deploy_resource_archive(self, resource_uri):
        raise NotImplementedError()


class LocalArtifact(Artifact):
    def __init__(self, filesystem, staging_directory: str, project=None):
        super().__init__(staging_directory, project)
        self._fs = filesystem

    def deploy_job_scripts(self, job_name, job_script, array_wrapper):
        def deploy_file(path, content):
            with self._fs.open(path, "wt") as f:
                f.write(content)

        job_path = self.job_path(job_name)
        job_log_path = os.path.join(job_path, "logs")

        self._fs.makedirs(job_path, exist_ok=True)
        self._fs.makedirs(job_log_path, exist_ok=True)
        deploy_file(self.job_script_path(job_name), job_script)
        deploy_file(self.job_array_wrapper_path(job_name), array_wrapper)
        console.log(f"Created job script {self.job_script_path(job_name)}")

    def deploy_singularity_container(self, singularity_image):
        fs = self._fs

        image_name = os.path.basename(singularity_image)
        deploy_container_path = self.singularity_image_path(image_name)
        if not fs.exists(deploy_container_path):
            console.log(f"Uploading container {image_name}...")
            fs.makedirs(os.path.dirname(deploy_container_path), exist_ok=True)
            fs.put(singularity_image, deploy_container_path)
            console.log(f"Deployed Singularity container to {deploy_container_path}")
        else:
            if fs.modified(deploy_container_path) < datetime.datetime.fromtimestamp(
                os.path.getmtime(singularity_image)
            ):
                console.log(
                    f"Local container is newer. Uploading container {image_name}..."
                )
                fs.put(singularity_image, deploy_container_path)
                console.log(
                    f"Deployed Singularity container to {deploy_container_path}"
                )
            else:
                console.log(f"Container {deploy_container_path} exists, skip upload.")
        return deploy_container_path

    def deploy_resource_archive(self, resource_uri):
        deploy_archive_path = self.archive_path(resource_uri)

        if not self._fs.exists(deploy_archive_path):
            self._fs.makedirs(os.path.dirname(deploy_archive_path), exist_ok=True)
            self._fs.put_file(resource_uri, deploy_archive_path)
            console.log(f"Deployed archive to {deploy_archive_path}")
        else:
            console.log(f"Archive {deploy_archive_path} exists, skipping upload.")
        return deploy_archive_path


class RemoteArtifact(Artifact):
    def __init__(self, hostname, user, staging_directory: str, project=None):
        fs = fsspec.filesystem("sftp", host=hostname, username=user)
        # Normalize the storage root to an absolute path.
        self._host = hostname
        self._user = user
        if not os.path.isabs(staging_directory):
            staging_directory = fs.ftp.normalize(staging_directory)
        super().__init__(staging_directory, project)
        self._fs = fs

    def deploy_job_scripts(self, job_name, job_script, array_wrapper):
        def deploy_file(path, content):
            with self._fs.open(path, "wt") as f:
                f.write(content)

        job_path = self.job_path(job_name)
        job_log_path = os.path.join(job_path, "logs")

        self._fs.makedirs(job_path, exist_ok=True)
        self._fs.makedirs(job_log_path, exist_ok=True)
        deploy_file(self.job_script_path(job_name), job_script)
        deploy_file(self.job_array_wrapper_path(job_name), array_wrapper)
        console.log(f"Created job script {self.job_script_path(job_name)}")

    def deploy_singularity_container(self, singularity_image):
        fs = self._fs

        image_name = os.path.basename(singularity_image)
        deploy_container_path = self.singularity_image_path(image_name)
        if not fs.exists(deploy_container_path):
            console.log(f"Uploading container {singularity_image}...")
            fs.makedirs(os.path.dirname(deploy_container_path), exist_ok=True)
            rsync(
                singularity_image,
                deploy_container_path,
                opt=["--info=progress2", "-havz"],
                host=f"{self._user}@{self._host}",
            )
            console.log(f"Deployed Singularity container to {deploy_container_path}")
        else:
            if fs.modified(deploy_container_path) < datetime.datetime.fromtimestamp(
                os.path.getmtime(singularity_image)
            ):
                console.log(
                    f"Local container is newer. Uploading container {image_name}..."
                )
                rsync(
                    singularity_image,
                    deploy_container_path,
                    opt=["--info=progress2", "-havz"],
                    host=f"{self._user}@{self._host}",
                )
                console.log(
                    f"Deployed Singularity container to {deploy_container_path}"
                )
            else:
                console.log(
                    f"Container {deploy_container_path} exists, skipping upload."
                )
        return deploy_container_path

    def deploy_resource_archive(self, resource_uri):
        deploy_archive_path = self.archive_path(resource_uri)

        if not self._fs.exists(deploy_archive_path):
            self._fs.makedirs(os.path.dirname(deploy_archive_path), exist_ok=True)
            self._fs.put_file(resource_uri, deploy_archive_path)
            console.log(f"Deployed archive to {deploy_archive_path}")
        else:
            console.log(f"Archive {deploy_archive_path} exists, skipping upload.")
        return deploy_archive_path
