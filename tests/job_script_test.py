from absl.testing import absltest
from absl.testing import parameterized

from lxm3.xm_cluster.execution import job_script


class ArrayWrapperTest(parameterized.TestCase):
    def test_env_vars(self):
        env_var_str = job_script._create_env_vars([{"FOO": "BAR1"}, {"FOO": "BAR2"}])
        expected = """\
FOO_0="BAR1"
FOO_1="BAR2"
FOO=$(eval echo \\$"FOO_$1")
export FOO"""
        self.assertEqual(env_var_str, expected)

    def test_empty_env_vars(self):
        self.assertEqual(job_script._create_env_vars([{}]), ":;")

    def test_different_keys(self):
        with self.assertRaises(ValueError):
            job_script._create_env_vars([{"FOO": "BAR1"}, {"BAR": "BAR2"}])

    def test_args(self):
        args_str = job_script._create_args(
            [["--seed=1", "--task=1"], ["--seed=2", "--task=2"]]
        )
        expected = """\
TASK_CMD_ARGS_0="--seed=1 --task=1"
TASK_CMD_ARGS_1="--seed=2 --task=2"
TASK_CMD_ARGS=$(eval echo \\$"TASK_CMD_ARGS_$1")
echo $TASK_CMD_ARGS"""
        self.assertEqual(args_str, expected)

    def test_empty_args(self):
        self.assertEqual(job_script._create_args([]), ":;")
        self.assertEqual(
            job_script._create_args([[]]),
            """\
TASK_CMD_ARGS_0=""
TASK_CMD_ARGS=$(eval echo \\$"TASK_CMD_ARGS_$1")
echo $TASK_CMD_ARGS""",
        )


if __name__ == "__main__":
    absltest.main()
