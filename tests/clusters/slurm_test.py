from unittest import mock

import fabric
from absl.testing import absltest
from absl.testing import parameterized

from lxm3.clusters import slurm


class SlurmTest(parameterized.TestCase):
    @parameterized.named_parameters(
        [
            {
                "testcase_name": "job",
                "text": "Submitted batch job 6",  # noqa
                "expected": 6,
            },
        ]
    )
    def test_parse_job_id(self, text, expected):
        job_id = slurm.parse_job_id(text)
        self.assertEqual(job_id, expected)


class ClientTest(absltest.TestCase):
    @mock.patch("fabric.Connection")
    def test_client(self, mock_connection):
        instance = mock_connection.return_value
        instance.run.return_value = fabric.Result(
            connection=instance,
            stdout="Submitted batch job 6",
        )
        client = slurm.Client(hostname="host", username="user")
        job_id = client.launch("job.sbatch")
        self.assertEqual(job_id, 6)
        client.close()


if __name__ == "__main__":
    absltest.main()
