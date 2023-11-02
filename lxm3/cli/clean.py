import datetime
import os
from typing import Optional

import fsspec
from rich.prompt import Confirm

from lxm3.xm_cluster import config as config_lib
from lxm3.xm_cluster.console import console

_JOB = "job"
_ARCHIVE = "archive"
_CONTAINER = "container"
_VALID_TYPES = [_JOB, _ARCHIVE, _CONTAINER]


class ClusterJob:
    def __init__(self, client, job_root) -> None:
        self._client = client
        self._job_root = job_root

    @property
    def job_name(self):
        return os.path.basename(self._job_root)

    def job_script(self):
        return (
            self._client._fs.cat(os.path.join(self._job_root, "job.sh"))
            .decode("utf-8")
            .strip()
        )

    def job_id(self):
        return (
            self._client._fs.cat(os.path.join(self._job_root, "job_id"))
            .decode("utf-8")
            .strip()
        )

    def logs(self):
        logs = self._client._fs.ls(os.path.join(self._job_root, "logs"))
        return logs

    def time_created(self):
        return self._client._fs.info(os.path.join(self._job_root, "job_id"))["mtime"]


class JobClient:
    def __init__(self, cluster_settings) -> None:
        self._cluster_settings = cluster_settings
        self._fs = fsspec.filesystem(
            "sftp",
            host=cluster_settings.hostname,
            username=cluster_settings.user,
            **cluster_settings.ssh_config,
        )

    @property
    def filesystem(self):
        return self._fs

    def list_projects(self):
        project_root = os.path.join(self._cluster_settings.storage_root, "projects")
        return [os.path.basename(p) for p in self._fs.ls(project_root)]

    def list_jobs(self, project):
        job_root = os.path.join(
            self._cluster_settings.storage_root, "projects", project, "jobs"
        )
        return [os.path.basename(p) for p in self._fs.ls(job_root)]

    def list_containers(self, project):
        container_root = os.path.join(
            self._cluster_settings.storage_root, "projects", project, "containers"
        )
        return [p for p in self._fs.ls(container_root)]

    def list_archives(self, project):
        archive_root = os.path.join(
            os.path.join(
                self._cluster_settings.storage_root, "projects", project, "archives"
            )
        )
        return [p for p in self._fs.ls(archive_root)]

    def get_job(self, project, job_name):
        job_path = os.path.join(
            self._cluster_settings.storage_root, "projects", project, "jobs", job_name
        )

        return ClusterJob(self, job_path)


def run_clean(
    project: str,
    days: Optional[int],
    dry_run: bool = False,
    force: bool = False,
    type_: Optional[str] = None,
):
    config = config_lib.default()
    client = JobClient(config.cluster_settings())
    now = datetime.datetime.now(tz=datetime.timezone.utc)

    if type_ is None:
        types = _VALID_TYPES
    else:
        types = type_.split(",")
        for type_ in types:
            if type_ not in _VALID_TYPES:
                raise ValueError("Invalid type: {}".format(type_))

    valid_projects = client.list_projects()
    if project not in valid_projects:
        raise ValueError(
            "Invalid project: {}, available projects are {}".format(
                project, valid_projects
            )
        )
    items_to_remove = []
    if _JOB in types:
        for job_name in client.list_jobs(project):
            job = client.get_job(project, job_name)
            job_root = job._job_root
            time_created = client.filesystem.info(job_root)["mtime"]
            if not days or (now - time_created > datetime.timedelta(days=days)):
                items_to_remove.append(("job", job_root))

    if _ARCHIVE in types:
        for archive in client.list_archives(project):
            time_created = client.filesystem.info(archive)["mtime"]
            if not days or (now - time_created > datetime.timedelta(days=days)):
                items_to_remove.append(("archive", archive))

    if _CONTAINER in types:
        for container in client.list_containers(project):
            time_created = client.filesystem.info(container)["mtime"]
            if not days or (now - time_created > datetime.timedelta(days=days)):
                items_to_remove.append(("container", container))

    if dry_run:
        if len(items_to_remove) > 0:
            for item_type, path in items_to_remove:
                console.print("Removing {} {}".format(item_type, path))
        else:
            console.print("No items to remove")
            return
    else:
        remove = False
        if not force:
            if len(items_to_remove) == 0:
                console.print("No items to remove")
                return
            for item_type, path in items_to_remove:
                console.print("Would remove [bold]{}[bold] {}".format(item_type, path))
            try:
                remove = Confirm.ask("Do you wish to continue?")
            except KeyboardInterrupt:
                console.print("Aborting")
                return
        else:
            remove = True

        if remove:
            for item_type, path in items_to_remove:
                console.print("Removing [bold]{}[bold] {}".format(item_type, path))
                try:
                    client.filesystem.rm(path, recursive=True)
                except:  # noqa
                    console.print_exception()
                    console.print("Failed to remove {}".format(path))
