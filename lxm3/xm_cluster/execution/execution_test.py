from unittest.mock import patch
from absl.testing import absltest
from absl.testing import parameterized

from lxm3.xm_cluster.execution import gridengine
from lxm3.xm_cluster import executors
from lxm3.xm_cluster import JobRequirements


class ConfigTest(parameterized.TestCase):
    def test_singularity_options_local(self):
        executor = executors.Local(
            singularity_options=[],
        )
        with patch("shutil.which", return_value=None):
            self.assertEqual(gridengine._get_singulation_options(executor), "")
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            self.assertEqual(gridengine._get_singulation_options(executor), "--nv")

    def test_singularity_options_gridengine(self):
        executor = executors.GridEngine(
            requirements=JobRequirements(gpu=1),
            singularity_options=[])
        self.assertEqual(gridengine._get_singulation_options(executor), "--nv")

        executor = executors.GridEngine(
            parallel_environments={"gpu": 1},
            singularity_options=[])
        self.assertEqual(gridengine._get_singulation_options(executor), "--nv")

        executor = executors.GridEngine(
            singularity_options=[])
        self.assertEqual(gridengine._get_singulation_options(executor), "")



if __name__ == "__main__":
    absltest.main()
