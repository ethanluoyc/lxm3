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


class ConfigTest(parameterized.TestCase):
    def test_config(self):
        config = config_lib.Config.from_string(_SAMPLE_CONFIG)
        self.assertTrue(isinstance(config["clusters"], list))
        self.assertEqual(config["clusters"][0]["name"], "cs")


if __name__ == "__main__":
    absltest.main()
