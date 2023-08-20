import os
import pathlib
from unittest import mock

import fabric
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
        match = gridengine.parse_job_id(text)
        self.assertEqual(gridengine.split_job_ids(match), expected)

    def test_parse_job_2(self):
        text = 'Your job-array 9834008.1-5:1 ("job-1677559794797") has been submitted'
        match = gridengine.parse_job_id(text)
        self.assertEqual(match.group(0), "9834008.1-5:1")

    @parameterized.named_parameters(
        [
            {
                "testcase_name": "qstat1",
                "input": "qstat.xml",
                "expected": {"9830166": {"state": "pending"}},
            },
            {
                "testcase_name": "qstat2",
                "input": "qstat2.xml",
                "expected": {
                    "9830188.1": {"state": "pending"},
                    "9830188.2": {"state": "pending"},
                    "9830188.3": {"state": "pending"},
                    "9830188.4": {"state": "pending"},
                    "9830188.5": {"state": "pending"},
                },
            },
            {
                "testcase_name": "qstat3",
                "input": "qstat3.xml",
                "expected": {
                    "9830197.1": {"state": "running"},
                    "9830197.2": {"state": "running"},
                    "9830197.3": {"state": "running"},
                    "9830197.4": {"state": "running"},
                    "9830197.5": {"state": "running"},
                },
            },
            {
                "testcase_name": "qstat4",
                "input": "qstat4.xml",
                "expected": {
                    "9830420.1": {"state": "running"},
                    "9830420.2": {"state": "running"},
                    "9830421.1": {"state": "pending"},
                    "9830421.2": {"state": "pending"},
                },
            },
        ]
    )
    def test_parse_qstat_state(self, input, expected):
        xml_output = pathlib.Path(
            os.path.join(os.path.dirname(__file__), "testdata", input)
        ).read_text()
        infos = gridengine.parse_qstat(xml_output)
        self.assertEqual(infos, expected)


class AccountingTest(absltest.TestCase):
    def test_accounting(self):
        DATA1 = (
            pathlib.Path(__file__).parent.joinpath("testdata/qacct1.txt").read_text()
        )
        DATA2 = (
            pathlib.Path(__file__).parent.joinpath("testdata/qacct2.txt").read_text()
        )
        self.assertLen(gridengine.parse_accounting(DATA1), 2)
        self.assertLen(gridengine.parse_accounting(DATA2), 2)


class ClientTest(absltest.TestCase):
    @mock.patch("fabric.Connection")
    def test_client(self, mock_connection):
        instance = mock_connection.return_value
        instance.run.return_value = fabric.Result(
            connection=instance,
            stdout='Your job 9830196 ("MyTESTJOBNAME") has been submitted',
        )
        client = gridengine.Client(hostname="host", username="user")
        match = client.launch("job.qsub")
        self.assertEqual(match.group(0), "9830196")
        client.close()


if __name__ == "__main__":
    absltest.main()
