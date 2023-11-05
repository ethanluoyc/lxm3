from absl.testing import absltest
from absl.testing import parameterized

from lxm3 import xm
from lxm3 import xm_cluster
from lxm3._vendor.xmanager import xm_mock


class ArrayJobTest(parameterized.TestCase):
    @parameterized.parameters(
        (1,),
        (5,),
    )
    def test_array_job_empty_env_vars(self, num_tasks):
        args = [{"seed": 1}] * num_tasks
        job = xm_cluster.ArrayJob(
            xm_mock.MockExecutable(), xm_mock.MockExecutor(), args=args
        )
        self.assertEqual(job.args, list(map(xm.SequentialArgs.from_collection, args)))
        self.assertEqual(job.env_vars, [{}] * num_tasks)

    @parameterized.parameters(
        (1,),
        (5,),
    )
    def test_array_job_empty_args(self, num_tasks):
        env_vars = [{"TASK": "foo"}] * num_tasks
        job = xm_cluster.ArrayJob(
            xm_mock.MockExecutable(), xm_mock.MockExecutor(), env_vars=env_vars
        )
        self.assertEqual(job.args, [xm.SequentialArgs()] * num_tasks)
        self.assertEqual(job.env_vars, env_vars)

    def test_broadcast_args(self):
        args = [{"seed": 1}, {"seed": 2}]
        env_vars = [{"TASK": "foo"}]
        job = xm_cluster.ArrayJob(
            xm_mock.MockExecutable(),
            xm_mock.MockExecutor(),
            args=args,
            env_vars=env_vars,
        )
        self.assertEqual(
            list(map(lambda x: x.to_list(), job.args)), [["--seed=1"], ["--seed=2"]]
        )
        self.assertEqual(job.env_vars, env_vars * 2)

    def test_broadcast_env_vars(self):
        args = [{"seed": 1}]
        env_vars = [{"TASK": "foo"}, {"TASK": "bar"}]
        job = xm_cluster.ArrayJob(
            xm_mock.MockExecutable(),
            xm_mock.MockExecutor(),
            args=args,
            env_vars=env_vars,
        )
        self.assertEqual(
            list(map(lambda x: x.to_list(), job.args)), [["--seed=1"], ["--seed=1"]]
        )
        self.assertEqual(job.env_vars, env_vars)

    def test_invalid_broadcast_args(self):
        args = [{"seed": 1}, {"seed": 2}, {"seed": 3}]
        env_vars = [{"TASK": "foo"}, {"TASK": "bar"}]
        with self.assertRaises(ValueError):
            xm_cluster.ArrayJob(
                xm_mock.MockExecutable(),
                xm_mock.MockExecutor(),
                args=args,
                env_vars=env_vars,
            )


if __name__ == "__main__":
    absltest.main()
