import functools
import os
from collections import UserDict

import appdirs
import tomli
from absl import flags

LXM_CONFIG = flags.DEFINE_string(
    "lxm_config", os.environ.get("LXM_CONFIG"), "Path to lxm configuration."
)


class Config(UserDict):
    @classmethod
    def from_file(cls, path: str) -> "Config":
        with open(path, "rt") as f:
            config_dict = tomli.loads(f.read())
        return cls(config_dict)

    @classmethod
    def from_string(cls, content: str) -> "Config":
        config_dict = tomli.loads(content)
        return cls(config_dict)

    def project(self):
        project = os.environ.get("LXM_PROJECT", None)
        if project is not None:
            return project
        return self.data.get("project", None)

    def local_config(self):
        return self.data["local"]

    def cluster_config(self, location=None):
        if location is None:
            return self.data["clusters"][0]
        else:
            clusters = {cluster["name"]: cluster for cluster in self.data["clusters"]}
            if location not in clusters:
                raise ValueError("Unknown cluster")
            cluster = clusters[location]
            return cluster


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
    if os.path.exists(cwd_path):
        return Config.from_file(cwd_path)
    user_config_path = os.path.join(appdirs.user_config_dir("lxm"), "lxm.toml")
    if os.path.exists(user_config_path):
        return Config.from_file(user_config_path)
    else:
        raise ValueError("Unable to load Config")
