# type: ignore
import unittest.mock

from absl.testing import absltest
from absl.testing import parameterized

from lxm3.xm_cluster import config as config_lib

_SAMPLE_CONFIG = """
project = ""
[local]
[local.storage]
staging = ".lxm"

[[clusters]]
name = "cs"
server = "beaker.cs.ucl.ac.uk"
user = "foo"

[clusters.storage]
staging = "/home/foo/lxm3-staging"

[[clusters]]
name = "myriad"
server = "myriad.rc.ucl.ac.uk"
user = "ucaby36"

[clusters.storage]
staging = "/home/bar/Scratch/lxm3-staging"

"""


def _test_config():
    return config_lib.Config.from_string(_SAMPLE_CONFIG)


class ConfigTest(parameterized.TestCase):
    def test_config(self):
        config = _test_config()
        self.assertTrue(isinstance(config._data["clusters"], list))
        self.assertEqual(config._data["clusters"][0]["name"], "cs")

    def test_local_config(self):
        config = _test_config()
        self.assertEqual(config.local_settings().storage_root, ".lxm")

    def test_default_cluster(self):
        config = _test_config()
        self.assertEqual(config.default_cluster(), "cs")
        with unittest.mock.patch.dict("os.environ", {"LXM_CLUSTER": "myriad"}):
            self.assertEqual(config.default_cluster(), "myriad")

    def test_config_project(self):
        config = config_lib.Config.from_string("")
        self.assertEqual(config.project(), None)
        with unittest.mock.patch.dict("os.environ", {"LXM_PROJECT": "test"}):
            self.assertEqual(config.project(), "test")

    def test_local_settings(self):
        settings = config_lib.LocalSettings()
        self.assertEqual(settings.env, {})
        self.assertEqual(settings.singularity.env, {})
        self.assertEqual(settings.singularity.binds, {})

    def test_cluster_settings(self):
        settings = config_lib.ClusterSettings()
        self.assertEqual(settings.env, {})
        self.assertEqual(settings.singularity.env, {})
        self.assertEqual(settings.singularity.binds, {})


if __name__ == "__main__":
    absltest.main()
