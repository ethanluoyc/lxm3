from unittest.mock import patch
import os
import fsspec
from absl.testing import absltest
from absl.testing import parameterized

from lxm3.xm_cluster.execution import gridengine
from lxm3.xm_cluster.execution import artifacts
from lxm3.xm_cluster import executors
from lxm3.xm_cluster import executables as cluster_executables
from lxm3.xm_cluster import JobRequirements
from lxm3 import xm


class ConfigTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("cpu", None, []),
        ("gpu", "/usr/bin/nvidia-smi", ["--nv"]),
    )
    def test_singularity_options_local(self, nvidia_smi_path, expected):
        executor = executors.Local()
        with patch("shutil.which", return_value=nvidia_smi_path):
            self.assertEqual(gridengine._get_singulation_options(executor), expected)

    @parameterized.named_parameters(
        (
            "sge_gpu",
            executors.GridEngine(requirements=JobRequirements(gpu=1)),
            ["--nv"],
        ),
        (
            "sge_gpu_pe",
            executors.GridEngine(parallel_environments={"gpu": 1}),
            ["--nv"],
        ),
        ("sge_default", executors.GridEngine(), []),
    )
    def test_singularity_options_gridengine(self, executor, expected):
        self.assertEqual(gridengine._get_singulation_options(executor), expected)

    def test_deploy_job(self):
        staging_dir = self.create_tempdir(name="staging")

        archive_name = "archive.zip"
        container_name = "container.sif"
        archive = staging_dir.create_file(archive_name)
        container = staging_dir.create_file(container_name)

        deploy_dir = self.create_tempdir(name="deploy")

        executor = executors.GridEngine(
            singularity_container=container.full_path,
        )
        executable = cluster_executables.Command(
            name="test",
            entrypoint_command="echo hello",
            resource_uri=archive.full_path,
        )
        version = "1"

        fs = fsspec.filesystem("file")
        job = xm.Job(executable, executor, name="test")
        artifact = artifacts.LocalArtifact(fs, deploy_dir.full_path)
        gridengine.deploy_job_resources(artifact, [job], version=version)
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
