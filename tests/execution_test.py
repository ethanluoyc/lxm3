import os
from unittest.mock import patch

from absl.testing import absltest
from absl.testing import parameterized

from lxm3 import xm
from lxm3.xm_cluster import JobRequirements
from lxm3.xm_cluster import config as config_lib
from lxm3.xm_cluster import executables as cluster_executables
from lxm3.xm_cluster import executors
from lxm3.xm_cluster.execution import artifacts
from lxm3.xm_cluster.execution import common
from lxm3.xm_cluster.execution import gridengine
from lxm3.xm_cluster.execution import job_script
from lxm3.xm_cluster.execution import local
from lxm3.xm_cluster.execution import slurm

_TEST_CONFIG = config_lib.Config.from_string(
    """\
project = ""
[local]
[local.storage]
staging = ".lxm"

[[clusters]]
name = "cs"
server = "beaker.cs.ucl.ac.uk"
user = "foo"

[clusters.storage]
staging = "/home/foo/lxm3-staging"

[[clusters]]
name = "myriad"
server = "myriad.rc.ucl.ac.uk"
user = "ucaby36"

[clusters.storage]
staging = "/home/bar/Scratch/lxm3-staging"

"""
)


class ExecutionTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("cpu", None, False),
        ("gpu", "/usr/bin/nvidia-smi", True),
    )
    def test_local_infer_gpu_request(self, nvidia_smi_path, expected):
        executor = executors.Local()
        with patch("shutil.which", return_value=nvidia_smi_path):
            use_gpu = local._is_gpu_requested(executor)
            self.assertEqual(use_gpu, expected)

    @parameterized.parameters(
        (executors.GridEngine(requirements=JobRequirements(gpu=1)), True),
        (executors.GridEngine(parallel_environments={"gpu": 1}), True),
        (executors.GridEngine(), False),
    )
    def test_sge_infer_gpu_request(self, executor, expected):
        use_gpu = gridengine._is_gpu_requested(executor)
        self.assertEqual(use_gpu, expected)

    @parameterized.parameters(
        (executors.SingularityOptions(), False, []),
        (executors.SingularityOptions(), True, ["--nv"]),
        (
            executors.SingularityOptions(
                bind={"/host": "/container", "/host2": "/container2"}
            ),
            False,
            ["--bind", "/host:/container", "--bind", "/host2:/container2"],
        ),
    )
    def test_get_singularity_options(self, options, use_gpu, expected):
        result = job_script.get_singulation_options(options, use_gpu)
        self.assertEqual(result, expected)

    def test_get_cluster_settings(self):
        executable = cluster_executables.Command(
            name="test",
            entrypoint_command="./entrypoint",
            resource_uri="test",
            singularity_image="test",
        )
        storage_root, hostname, user = common.get_cluster_settings(
            _TEST_CONFIG, [xm.Job(executable, executors.GridEngine())]
        )
        self.assertEqual(storage_root, "/home/foo/lxm3-staging")
        self.assertEqual(hostname, "beaker.cs.ucl.ac.uk")
        self.assertEqual(user, "foo")

        with self.assertRaises(ValueError):
            jobs = [xm.Job(executable, executors.Local())]
            common.get_cluster_settings(_TEST_CONFIG, jobs)

        with self.assertRaises(ValueError):
            jobs = [
                xm.Job(
                    executable,
                    executors.GridEngine(
                        requirements=JobRequirements(location="myriad")
                    ),
                ),
                xm.Job(
                    executable,
                    executors.GridEngine(requirements=JobRequirements(location="cs")),
                ),
            ]
            common.get_cluster_settings(_TEST_CONFIG, jobs)

        with self.assertRaises(ValueError):
            jobs = [
                xm.Job(
                    executable,
                    executors.Slurm(requirements=JobRequirements(location="myriad")),
                ),
                xm.Job(
                    executable,
                    executors.GridEngine(requirements=JobRequirements(location="cs")),
                ),
            ]
            common.get_cluster_settings(_TEST_CONFIG, jobs)

    def test_gridengine_header(self):
        executor = executors.GridEngine(
            resources={"h_vmem": "1G"},
            parallel_environments={"gpu": 1},
            project="test",
            account="alloc",
            walltime=10 * xm.Min,
        )
        header = gridengine._generate_header_from_executor(
            "test_job", executor, None, "/logs"
        )
        self.assertIn("#$ -l h_vmem=1G", header)
        self.assertIn("#$ -l h_rt=00:10:00", header)
        self.assertIn("#$ -pe gpu 1", header)
        self.assertIn("#$ -P test", header)
        self.assertIn("#$ -A alloc", header)

        executor = executors.GridEngine(max_parallel_tasks=2)
        header = gridengine._generate_header_from_executor(
            "test_job", executor, 10, "/logs"
        )
        self.assertIn("#$ -t 1-10", header)
        self.assertIn("#$ -tc 2", header)

    def test_slurm_header(self):
        executor = executors.Slurm(
            resources={"mem": "1G"},
            walltime=10 * xm.Hr,
            exclusive=True,
            partition="contrib-gpu-long",
        )
        header = slurm._generate_header_from_executor(
            "test_job", executor, None, "/logs"
        )
        self.assertIn("#SBATCH --mem=1G", header)
        self.assertIn("#SBATCH --time=10:00:00", header)
        self.assertIn("#SBATCH --exclusive", header)
        self.assertIn("#SBATCH --partition=contrib-gpu-long", header)

        executor = executors.Slurm()
        header = slurm._generate_header_from_executor("test_job", executor, 10, "/logs")
        self.assertIn("#SBATCH --array=1-10", header)

    @parameterized.named_parameters(
        ("local", executors.Local(), local.deploy_job_resources),
        ("sge", executors.GridEngine(), gridengine.deploy_job_resources),
        ("slurm", executors.Slurm(), slurm.deploy_job_resources),
    )
    def test_deploy_job(self, executor, deploy_fn):
        staging_dir = self.create_tempdir(name="staging")

        archive_name = "archive.zip"
        container_name = "container.sif"
        archive = staging_dir.create_file(archive_name)
        container = staging_dir.create_file(container_name)

        deploy_dir = self.create_tempdir(name="deploy")
        executable = cluster_executables.Command(
            name="test",
            entrypoint_command="echo hello",
            resource_uri=archive.full_path,
            singularity_image=container.full_path,
        )
        version = "1"

        job = xm.Job(executable, executor, name="test")
        artifact = artifacts.LocalArtifact(deploy_dir.full_path)
        deploy_fn(artifact=artifact, jobs=[job], version=version)

        expected_paths = [
            f"containers/{container_name}",
            f"jobs/job-{version}/job.sh",
            f"jobs/job-{version}/array_wrapper.sh",
            f"archives/{archive_name}",
        ]
        for path in expected_paths:
            with self.subTest(path):
                full_path = os.path.exists(os.path.join(deploy_dir, path))
                self.assertTrue(full_path)


if __name__ == "__main__":
    absltest.main()
