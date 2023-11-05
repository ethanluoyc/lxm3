import functools
import os
from typing import Any, Dict, Optional, Protocol

import appdirs
import tomlkit
from absl import flags
from absl import logging

LXM_CONFIG = flags.DEFINE_string(
    "lxm_config", os.environ.get("LXM_CONFIG"), "Path to lxm configuration."
)


class SingularitySettings:
    def __init__(self, data) -> None:
        self._data = data

    def __repr__(self) -> str:
        return repr(self._data)

    @property
    def binds(self) -> Dict[str, str]:
        binds = self._data.get("binds", [])
        return {bind["src"]: bind["dest"] for bind in binds}

    @property
    def env(self) -> Dict[str, str]:
        return self._data.get("env", {})


class ExecutionSettings(Protocol):
    @property
    def env(self) -> Dict[str, str]:
        ...

    @property
    def singularity(self) -> SingularitySettings:
        ...


class LocalSettings(ExecutionSettings):
    def __init__(self, data=None) -> None:
        self._data = data or {}

    def __repr__(self) -> str:
        return repr(self._data)

    @property
    def storage_root(self) -> str:
        return self._data["storage"]["staging"]

    @property
    def env(self) -> Dict[str, str]:
        return self._data.get("env", {})

    @property
    def singularity(self) -> SingularitySettings:
        return SingularitySettings(self._data.get("singularity", {}))


class ClusterSettings(ExecutionSettings):
    def __init__(self, data=None) -> None:
        self._data = data or {}

    def __repr__(self) -> str:
        return repr(self._data)

    @property
    def storage_root(self):
        return self._data["storage"]["staging"]

    @property
    def hostname(self):
        return self._data.get("server", None)

    @property
    def user(self):
        return self._data.get("user", None)

    @property
    def ssh_config(self):
        connect_kwargs = {}
        proxycommand = self._data.get("proxycommand", None)
        if proxycommand is not None:
            import paramiko

            connect_kwargs["sock"] = paramiko.ProxyCommand(proxycommand)

        ssh_private_key = self._data.get("ssh_private_key", None)
        if ssh_private_key is not None:
            connect_kwargs["key_filename"] = os.path.expanduser(ssh_private_key)

        password = self._data.get("password", None)
        if password is not None:
            connect_kwargs["password"] = password

        return connect_kwargs

    @property
    def env(self) -> Dict[str, str]:
        return self._data.get("env", {})

    @property
    def singularity(self) -> SingularitySettings:
        return SingularitySettings(self._data.get("singularity", {}))


class Config:
    def __init__(self, data) -> None:
        self._data = data

    def __repr__(self) -> Any:
        return repr(self._data)

    @classmethod
    def from_file(cls, path: str) -> "Config":
        with open(path, "rt") as f:
            config_dict = tomlkit.loads(f.read())
        return cls(config_dict)

    @classmethod
    def from_string(cls, content: str) -> "Config":
        config_dict = tomlkit.loads(content)
        return cls(config_dict)

    def project(self) -> Optional[str]:
        project = os.environ.get("LXM_PROJECT", None)
        if project is not None:
            return project
        return self._data.get("project", None)

    def set_project(self, project):
        self._data["project"] = project

    def local_settings(self) -> LocalSettings:
        return LocalSettings(self._data["local"])

    def default_cluster(self) -> str:
        cluster = os.environ.get("LXM_CLUSTER", None)
        if cluster is None:
            cluster = self._data["clusters"][0]["name"]
        return cluster

    def cluster_settings(self) -> ClusterSettings:
        location = self.default_cluster()
        clusters = {cluster["name"]: cluster for cluster in self._data["clusters"]}
        if location not in clusters:
            raise ValueError("Unknown cluster")
        cluster_config = clusters[location]
        return ClusterSettings(cluster_config)


@functools.lru_cache()
def default() -> Config:
    # The path may be relative, especially if it comes from sys.argv[0].
    try:
        if LXM_CONFIG.value is not None:
            logging.debug("Loading config from %s", LXM_CONFIG.value)
            return Config.from_file(LXM_CONFIG.value)
    except flags.UnparsedFlagAccessError:
        logging.debug("Unable to load config from flag")
    # Use file from environment variable
    lxm_config_path = os.environ.get("LXM_CONFIG", None)
    if lxm_config_path is not None:
        logging.debug("Loading config from %s", lxm_config_path)
        return Config.from_file(lxm_config_path)
    cwd_path = os.path.join(os.getcwd(), "lxm.toml")
    if os.path.exists(cwd_path):
        logging.debug("Loading config from %s", cwd_path)
        return Config.from_file(cwd_path)
    user_config_path = os.path.join(appdirs.user_config_dir("lxm3"), "config.toml")
    if os.path.exists(user_config_path):
        logging.debug("Loading config from %s", cwd_path)
        return Config.from_file(user_config_path)
    else:
        raise ValueError("Unable to load Config")
