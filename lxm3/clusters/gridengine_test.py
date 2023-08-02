from absl.testing import absltest
from absl.testing import parameterized

from lxm3.clusters import gridengine


class GridEngineTest(parameterized.TestCase):
    @parameterized.named_parameters(
        [
            {
                "testcase_name": "array job",
                "text": 'Your job-array 9830192.1-5:1 ("MyTESTJOBNAME") has been submitted',  # noqa
                "expected": [
                    "9830192.1",
                    "9830192.2",
                    "9830192.3",
                    "9830192.4",
                    "9830192.5",
                ],
            },
            {
                "testcase_name": "job",
                "text": 'Your job 9830196 ("MyTESTJOBNAME") has been submitted',
                "expected": ["9830196"],
            },
            {
                "testcase_name": "job name with numbers",
                "text": 'Your job-array 9834008 ("job-1677559794797") has been submitted',  # noqa
                "expected": ["9834008"],
            },
            {
                "testcase_name": "array-job name with numbers",
                "text": 'Your job-array 9834008.1-5:1 ("job-1677559794797") has been submitted',  # noqa
                "expected": [f"9834008.{i}" for i in range(1, 6)],
            },
        ]
    )
    def test_parse_job_id(self, text, expected):
        match = gridengine._extract_job_id(text)
        self.assertEqual(gridengine._split_job_ids(match), expected)


if __name__ == "__main__":
    absltest.main()
