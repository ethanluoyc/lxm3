from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from lxm3 import xm
from lxm3 import xm_cluster
from lxm3.xm_cluster import array_job
from lxm3.xm_cluster import config as config_lib
from lxm3.xm_cluster import experiment as cluster_experiment

_TEST_CONFIG = """
project = ""
[local]
[local.storage]
staging = "{}"

[[clusters]]
name = "default"
server = "localhost"
user = "user"

[clusters.storage]
staging = "/home/foo/lxm3-staging"
"""


class DummyHandle:
    async def wait(self):
        return


def _fake_launch(job):
    print(job)
    if isinstance(job, xm_cluster.ArrayJob):
        jobs = array_job.flatten_array_job(job)
    else:
        jobs = xm.job_operators.flatten_jobs(job)

    return cluster_experiment._LaunchResult(
        local_handles=[DummyHandle()] * len(jobs), non_local_handles=[]
    )


class ExperimentTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        staging = self.create_tempdir().full_path
        self._config = config_lib.Config.from_string(_TEST_CONFIG.format(staging))
        self._executable = xm_cluster.Command(
            name="dummy",
            entrypoint_command="./entrypint.sh",
            resource_uri="dummy",
        )

    @mock.patch.object(cluster_experiment, "_launch")
    def test_create_experiment(self, mock_launch):
        with mock.patch.object(config_lib, "default") as mock_default:
            mock_default.return_value = self._config
            job = xm.Job(executable=self._executable, executor=xm_cluster.Local())
            jobs = [job] * 2
            mock_launch.return_value = _fake_launch(job)
            experiment = xm_cluster.create_experiment("test")
            with experiment:
                for job in jobs:
                    experiment.add(job)
            self.assertEqual(experiment.work_unit_count, 2)
            self.assertEqual(mock_launch.call_count, 2)

    @mock.patch.object(cluster_experiment, "_launch")
    def test_create_experiment_batch(self, mock_launch):
        with mock.patch.object(config_lib, "default") as mock_default:
            mock_default.return_value = self._config
            job = xm_cluster.ArrayJob(
                executable=self._executable,
                executor=xm_cluster.Local(),
                args=[{"seed": 1}, {"seed": 2}],
            )
            mock_launch.return_value = _fake_launch(job)
            experiment = xm_cluster.create_experiment("test")
            with experiment:
                experiment.add(job)
            self.assertEqual(experiment.work_unit_count, 1)
            mock_launch.assert_called_once()


if __name__ == "__main__":
    absltest.main()
