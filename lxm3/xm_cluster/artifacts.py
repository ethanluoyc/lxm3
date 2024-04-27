import datetime
import functools
import os
import shutil
from typing import Optional, Tuple

import attr
import fsspec
from fsspec.implementations import local
from fsspec.implementations import sftp

from lxm3.xm_cluster import config as config_lib


@attr.s(auto_attribs=True)
class FileInfo:
    path: str
    size: int
    time_modified: datetime.datetime


class ArtifactStore:
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
        self._filesystem = filesystem
        self._initialize()

    @property
    def filesystem(self):
        return self._filesystem

    @property
    def storage_root(self):
        return self._storage_root

    def _initialize(self):
        self.ensure_dir(".")
        if not self.exists(".gitignore"):
            self.put_text("*\n", ".gitignore")

    def get_file_info(self, path: str) -> FileInfo:
        path = self.normalize_path(path)
        info = self._filesystem.info(path)
        size = info["size"]
        mod_time = info["mtime"]
        if not isinstance(mod_time, datetime.datetime):
            mod_time = datetime.datetime.fromtimestamp(
                info["mtime"], tz=datetime.timezone.utc
            )
        return FileInfo(path, size, mod_time)

    def normalize_path(self, path: str) -> str:
        assert not os.path.isabs(path)
        return os.path.join(self._storage_root, path)

    def exists(self, name: str) -> bool:
        return self._filesystem.exists(self.normalize_path(name))

    def ensure_dir(self, directory: str):
        directory = self.normalize_path(directory)
        return self._filesystem.makedirs(directory, exist_ok=True)

    def put_file(self, lpath: str, rpath: str) -> str:
        path = self.normalize_path(rpath)

        self._filesystem.makedirs(os.path.dirname(path), exist_ok=True)
        self._filesystem.put(lpath, path)

        return path

    def put_text(self, text: str, rpath: str) -> str:
        path = self.normalize_path(rpath)

        self._filesystem.makedirs(os.path.dirname(path), exist_ok=True)
        self._filesystem.write_text(path, text)

        return path

    def put_fileobj(self, fileobj, rpath: str) -> str:
        path = self.normalize_path(rpath)

        self._filesystem.makedirs(os.path.dirname(path), exist_ok=True)

        filesystem = self._filesystem
        if isinstance(filesystem, local.LocalFileSystem):
            with filesystem.open(path, "wb") as fout:
                shutil.copyfileobj(fileobj, fout)
        elif isinstance(filesystem, sftp.SFTPFileSystem):
            filesystem.ftp.putfo(fileobj, path)
        else:
            raise NotImplementedError()

        return path

    def should_update(self, lpath: str, rpath: str) -> Tuple[bool, str]:
        if not self.exists(rpath):
            return True, "file does not exist"

        local_stat = os.stat(lpath)
        local_size = local_stat.st_size
        local_mtime = datetime.datetime.fromtimestamp(
            local_stat.st_atime, tz=datetime.timezone.utc
        )
        remote_file_info = self.get_file_info(rpath)

        if local_size != remote_file_info.size:
            return True, "file size mismatch"

        if local_mtime > remote_file_info.time_modified:
            return True, "local file is newer"

        return False, ""


@functools.lru_cache(maxsize=None)
def get_local_artifact_store() -> ArtifactStore:
    default = config_lib.default()
    settings = default.local_settings()
    project = default.project()
    filesystem = fsspec.filesystem("file")
    storage_root = os.path.abspath(os.path.expanduser(settings.storage_root))
    return ArtifactStore(filesystem, storage_root, project=project)


@functools.lru_cache(maxsize=None)
def get_cluster_artifact_store() -> ArtifactStore:
    default = config_lib.default()
    settings = default.cluster_settings()
    project = default.project()

    if settings.hostname is None:
        filesystem = fsspec.filesystem("file")
        storage_root = os.path.abspath(os.path.expanduser(settings.storage_root))
    else:
        filesystem = sftp.SFTPFileSystem(
            host=settings.hostname, username=settings.user, **settings.ssh_config
        )
        storage_root = filesystem.ftp.normalize(settings.storage_root)

    return ArtifactStore(filesystem, staging_directory=storage_root, project=project)
