import functools
import os
from collections import UserDict
from typing import Any, Dict, Optional

import appdirs
import tomlkit
from absl import flags

LXM_CONFIG = flags.DEFINE_string(
    "lxm_config", os.environ.get("LXM_CONFIG"), "Path to lxm configuration."
)


class Config(UserDict):
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
        return self.data.get("project", None)

    def local_config(self) -> Dict[str, Any]:
        return self.data["local"]

    def default_cluster(self) -> str:
        cluster = os.environ.get("LXM_CLUSTER", None)
        if cluster is None:
            cluster = self.data["clusters"][0]["name"]
        return cluster

    def get_cluster_settings(self):
        location = self.default_cluster()
        clusters = {cluster["name"]: cluster for cluster in self.data["clusters"]}
        if location not in clusters:
            raise ValueError("Unknown cluster")
        cluster_config = clusters[location]
        storage_root = cluster_config["storage"]["staging"]
        hostname = cluster_config.get("server", None)
        user = cluster_config.get("user", None)

        connect_kwargs = {}

        proxycommand = cluster_config.get("proxycommand", None)
        if proxycommand is not None:
            import paramiko

            connect_kwargs["sock"] = paramiko.ProxyCommand(proxycommand)

        ssh_private_key = cluster_config.get("ssh_private_key", None)
        if ssh_private_key is not None:
            connect_kwargs["key_filename"] = os.path.expanduser(ssh_private_key)

        password = cluster_config.get("password", None)
        if password is not None:
            connect_kwargs["password"] = password

        return storage_root, hostname, user, connect_kwargs


@functools.lru_cache()
def default() -> Config:
    # The path may be relative, especially if it comes from sys.argv[0].
    if LXM_CONFIG.value is not None:
        return Config.from_file(LXM_CONFIG.value)
    # Use file from environment variable
    lxm_config_path = os.environ.get("LXM_CONFIG", None)
    if lxm_config_path is not None:
        return Config.from_file(lxm_config_path)
    cwd_path = os.path.join(os.getcwd(), "lxm.toml")
    if os.path.exists(cwd_path):
        return Config.from_file(cwd_path)
    user_config_path = os.path.join(appdirs.user_config_dir("lxm3"), "config.toml")
    if os.path.exists(user_config_path):
        return Config.from_file(user_config_path)
    else:
        raise ValueError("Unable to load Config")
