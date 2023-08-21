import datetime

from absl.testing import absltest
from absl.testing import parameterized

from lxm3.xm_cluster import executors
from lxm3.xm_cluster.execution import gridengine
from lxm3.xm_cluster.execution import slurm


class ExecutorTest(parameterized.TestCase):
    @parameterized.parameters(
        (None, None),
        (5 * 3600 + 20 * 60 + 25, datetime.timedelta(hours=5, minutes=20, seconds=25)),
        (datetime.timedelta(hours=3), datetime.timedelta(hours=3)),
    )
    def test_convert_time(self, input, expected):
        self.assertEqual(executors._convert_time(input), expected)

    @parameterized.parameters(
        (datetime.timedelta(hours=5, minutes=20, seconds=25), "05:20:25"),
        (datetime.timedelta(hours=55, minutes=20, seconds=25), "55:20:25"),
        (datetime.timedelta(hours=102, minutes=20, seconds=25), "102:20:25"),
    )
    def test_format_sge_time(self, input, expected):
        self.assertEqual(gridengine._format_time(input.total_seconds()), expected)

    @parameterized.parameters(
        (datetime.timedelta(hours=5, minutes=20, seconds=25), "05:20:25"),
        (datetime.timedelta(hours=55, minutes=20, seconds=25), "02-07:20:25"),
    )
    def test_format_slurm_time(self, input, expected):
        self.assertEqual(slurm._format_slurm_time(input), expected)


if __name__ == "__main__":
    absltest.main()
