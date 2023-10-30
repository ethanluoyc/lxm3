import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from lxm3 import xm
from lxm3.contrib import ucl
from lxm3.xm_cluster import requirements as cluster_requirements


class UCLClusterTest(parameterized.TestCase):
    @parameterized.named_parameters(
        (
            "gpu",
            cluster_requirements.JobRequirements(gpu=1, ram=8 * xm.GB),
            {"gpu": "true", "tmem": 8 * xm.GB},
            {},
        ),
        (
            "gpu_2",
            cluster_requirements.JobRequirements(gpu=2, ram=8 * xm.GB),
            {"gpu": "true", "tmem": 8 * xm.GB},
            {"gpu": 2},
        ),
        (
            "cpu",
            cluster_requirements.JobRequirements(ram=8 * xm.GB),
            {"tmem": 8 * xm.GB, "h_vmem": 8 * xm.GB},
            {},
        ),
        (
            "cpu_gpu_0",
            cluster_requirements.JobRequirements(gpu=0, ram=8 * xm.GB),
            {"tmem": 8 * xm.GB, "h_vmem": 8 * xm.GB},
            {},
        ),
        (
            "cpu_2",
            cluster_requirements.JobRequirements(gpu=0, cpu=2, ram=8 * xm.GB),
            {"tmem": 8 * xm.GB, "h_vmem": 8 * xm.GB},
            {"smp": 2},
        ),
    )
    def test_cs_cluster(self, requirements, expected_resources, expected_pe):
        with mock.patch.dict(os.environ, {"LXM_CLUSTER": "cs"}):
            executor = ucl.UclGridEngine(requirements)
            self.assertEqual(executor.resources, expected_resources)
            self.assertEqual(executor.parallel_environments, expected_pe)

    @parameterized.named_parameters(
        (
            "gpu",
            cluster_requirements.JobRequirements(gpu=1, ram=8 * xm.GB),
            {"gpu": 1, "mem": 8 * xm.GB},
            {},
        ),
        (
            "gpu_2",
            cluster_requirements.JobRequirements(gpu=2, ram=8 * xm.GB),
            {"gpu": 2, "mem": 8 * xm.GB},
            {},
        ),
        (
            "cpu",
            cluster_requirements.JobRequirements(ram=8 * xm.GB),
            {"mem": 8 * xm.GB},
            {},
        ),
        (
            "cpu_gpu_0",
            cluster_requirements.JobRequirements(gpu=0, ram=8 * xm.GB),
            {"mem": 8 * xm.GB},
            {},
        ),
        (
            "cpu_2",
            cluster_requirements.JobRequirements(gpu=0, cpu=2, ram=8 * xm.GB),
            {"mem": 8 * xm.GB},
            {"smp": 2},
        ),
    )
    def test_myriad_cluster(self, requirements, expected_resources, expected_pe):
        with mock.patch.dict(os.environ, {"LXM_CLUSTER": "myriad"}):
            executor = ucl.UclGridEngine(requirements)
            self.assertEqual(executor.resources, expected_resources)
            self.assertEqual(executor.parallel_environments, expected_pe)

    def test_invalid_location(self):
        requirements = cluster_requirements.JobRequirements(location="unknown")
        with self.assertRaises(ValueError):
            ucl.UclGridEngine(requirements)


if __name__ == "__main__":
    absltest.main()
