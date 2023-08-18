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


if __name__ == "__main__":
    absltest.main()
