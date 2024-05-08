import os
import subprocess
import textwrap
import unittest
import zipfile
from unittest import mock
from unittest.mock import patch

import fsspec
import pytest
from absl.testing import absltest
from absl.testing import parameterized

from lxm3 import xm
from lxm3 import xm_cluster
from lxm3.clusters import gridengine as gridengine_cluster
from lxm3.clusters import slurm as slurm_cluster
from lxm3.xm_cluster import JobRequirements
from lxm3.xm_cluster import artifacts
from lxm3.xm_cluster import config
from lxm3.xm_cluster import executables
from lxm3.xm_cluster import executors
from lxm3.xm_cluster.execution import gridengine
from lxm3.xm_cluster.execution import job_script_builder as job_script
from lxm3.xm_cluster.execution import local
from lxm3.xm_cluster.execution import slurm
from tests import utils


def _create_singularity_image(name):
    return executables.ContainerImage(
        name=name, image_type=executables.ContainerImageType.SINGULARITY
    )


def _create_docker_image(name):
    return executables.ContainerImage(
        name=name, image_type=executables.ContainerImageType.DOCKER
    )


class JobScriptBuilderTest(parameterized.TestCase):
    def test_env_vars(self):
        env_var_str = job_script._create_env_vars(
            [{"FOO": "BAR1"}, {"FOO": "BAR2"}], "LXM_TASK_ID", 0
        )
        expected = textwrap.dedent("""\
            FOO_0="BAR1"
            FOO_1="BAR2"
            FOO=$(eval echo \\$"FOO_$LXM_TASK_ID")
            export FOO""")
        self.assertEqual(env_var_str, expected)

    def test_empty_env_vars(self):
        self.assertEqual(job_script._create_env_vars([{}], "LXM_TASK_ID", 0), "")

    def test_common_values(self):
        env_var_str = job_script._create_env_vars(
            [{"FOO": "BAR", "BAR": "1"}, {"FOO": "BAR", "BAR": "2"}], "LXM_TASK_ID", 0
        )
        expected = textwrap.dedent("""\
            export FOO="BAR"
            BAR_0="1"
            BAR_1="2"
            BAR=$(eval echo \\$"BAR_$LXM_TASK_ID")
            export BAR""")
        self.assertEqual(env_var_str, expected)

    def test_different_keys(self):
        with self.assertRaises(ValueError):
            job_script._create_env_vars(
                [{"FOO": "BAR1"}, {"BAR": "BAR2"}], "LXM_TASK_ID", 0
            )

    def test_args(self):
        args_str = job_script._create_args(
            [["--seed=1", "--task=1"], ["--seed=2", "--task=2"]], "LXM_TASK_ID", 0
        )
        expected = textwrap.dedent("""\
            TASK_CMD_ARGS_0="--seed=1 --task=1"
            TASK_CMD_ARGS_1="--seed=2 --task=2"
            TASK_CMD_ARGS=$(eval echo \\$"TASK_CMD_ARGS_$LXM_TASK_ID")
            eval set -- $TASK_CMD_ARGS""")
        self.assertEqual(args_str, expected)

    def test_empty_args(self):
        self.assertEqual(job_script._create_args([], "LXM_TASK_ID", 0), "")
        self.assertEqual(
            job_script._create_args([[]], "LXM_TASK_ID", 0),
            textwrap.dedent(
                """\
                TASK_CMD_ARGS_0=""
                TASK_CMD_ARGS=$(eval echo \\$"TASK_CMD_ARGS_$LXM_TASK_ID")
                eval set -- $TASK_CMD_ARGS""",
            ),
        )

    def test_ml_collections_quoting(self):
        args = xm.SequentialArgs.from_collection({"config.name": "train[:90%]"})
        args = args.to_list()
        self.assertEqual(
            job_script._create_args([args], "LXM_TASK_ID", 0),
            textwrap.dedent(
                """\
                TASK_CMD_ARGS_0="--config.name='train[:90%]'"
                TASK_CMD_ARGS=$(eval echo \\$"TASK_CMD_ARGS_$LXM_TASK_ID")
                eval set -- $TASK_CMD_ARGS""",
            ),
        )


class LocalExecutionTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("cpu", None, False),
        ("gpu", "/usr/bin/nvidia-smi", True),
    )
    def test_local_infer_gpu_request(self, nvidia_smi_path, expected):
        executor = executors.Local()
        with patch("shutil.which", return_value=nvidia_smi_path):
            use_gpu = local.LocalJobScriptBuilder._is_gpu_requested(executor)
            self.assertEqual(use_gpu, expected)

    def test_local_launch(self):
        staging_dir = self.create_tempdir(name="staging")

        archive_name = "archive.zip"
        container_name = "container.sif"
        archive = staging_dir.create_file(archive_name)
        container = staging_dir.create_file(container_name)

        deploy_dir = self.create_tempdir(name="deploy")
        executable = executables.AppBundle(
            name="test",
            entrypoint_command="echo hello",
            resource_uri=archive.full_path,
            container_image=_create_singularity_image(container.full_path),
        )
        executor = executors.Local()
        job = xm.Job(executable, executor, name="test")
        artifact = artifacts.ArtifactStore(
            fsspec.filesystem("local"), deploy_dir.full_path
        )
        settings = config.LocalSettings({})
        client = local.LocalClient(settings, artifact)
        with mock.patch.object(subprocess, "run"):
            client.launch("test_job", job)

    def _run_job_script(self, job_script_content, env=None):
        job_script = self.create_tempfile("job.sh", content=job_script_content)
        os.chmod(job_script.full_path, 0o755)
        workdir = self.create_tempdir("workdir").full_path

        try:
            process = subprocess.run(
                [job_script.full_path],
                check=True,
                env=env,
                capture_output=True,
                cwd=workdir,
            )
        except subprocess.CalledProcessError as e:
            print("job_script:", job_script_content)
            print("stdout:", e.stdout.decode("utf-8"))
            print("stderr:", e.stderr.decode("utf-8"))
            raise
        return process

    def _create_bundle(self, entrypoint: str, container_image=None):
        tmpf = self.create_tempfile("test.zip")
        with zipfile.ZipFile(tmpf.full_path, "w") as z:
            info = zipfile.ZipInfo("entrypoint.sh")
            info.external_attr = 0o777 << 16  # give full access to included file
            z.writestr(info, entrypoint)

        return xm_cluster.AppBundle(
            "foo",
            "./entrypoint.sh",
            resource_uri=tmpf.full_path,
            container_image=container_image,
        )

    def test_job_script_run_single_job(self):
        executable = self._create_bundle("#!/usr/bin/env bash\necho $@ $FOO")
        job = xm.Job(
            executable,
            executor=xm_cluster.Local(),
            args={"seed": 1},
            env_vars={"FOO": "FOO_0"},
        )
        builder = local.LocalJobScriptBuilder()
        job_script_content = builder.build(job, "foo", "/tmp")
        process = self._run_job_script(job_script_content)
        self.assertEqual(process.stdout.decode("utf-8").strip(), "--seed=1 FOO_0")

    def test_job_script_run_array_job(self):
        executable = self._create_bundle("#!/usr/bin/env bash\necho $@ $FOO")
        job = xm_cluster.ArrayJob(
            executable,
            executor=xm_cluster.Local(),
            args=[{"seed": 1}, {"seed": 2}],
            env_vars=[{"FOO": "FOO_0"}, {"FOO": "FOO_1"}],
        )
        builder = local.LocalJobScriptBuilder()
        job_script_content = builder.build(job, "foo", "/tmp")
        process = self._run_job_script(
            job_script_content, env={builder.ARRAY_TASK_ID: "1"}
        )
        self.assertEqual(process.stdout.decode("utf-8").strip(), "--seed=1 FOO_0")
        process = self._run_job_script(
            job_script_content, env={builder.ARRAY_TASK_ID: "2"}
        )
        self.assertEqual(process.stdout.decode("utf-8").strip(), "--seed=2 FOO_1")

    @pytest.mark.integration
    @unittest.skipIf(
        not utils.is_singularity_installed(), "Singularity is not installed"
    )
    def test_singularity(self):
        executable = self._create_bundle(
            "#!/usr/bin/env bash\necho $FOO",
            container_image=_create_singularity_image("docker://python:3.10-slim"),
        )
        job = xm_cluster.ArrayJob(
            executable,
            executor=xm_cluster.Local(
                singularity_options=xm_cluster.SingularityOptions(
                    extra_options=["--containall"]
                )
            ),
            env_vars=[{"FOO": "FOO_0"}],
        )
        job_script_content = local.LocalJobScriptBuilder().build(job, "foo", "/tmp")
        process = self._run_job_script(
            job_script_content, env={local.LocalJobScriptBuilder.ARRAY_TASK_ID: "1"}
        )
        self.assertEqual(process.stdout.decode("utf-8").strip(), "FOO_0")

    @pytest.mark.integration
    @unittest.skipIf(not utils.is_docker_installed(), "Docker is not installed")
    def test_docker_image(self):
        executable = self._create_bundle(
            "#!/usr/bin/env bash\necho $FOO",
            container_image=_create_docker_image("python:3.10-slim"),
        )
        job = xm_cluster.ArrayJob(
            executable, executor=xm_cluster.Local(), env_vars=[{"FOO": "FOO_0"}]
        )
        job_script_content = local.LocalJobScriptBuilder().build(job, "foo", "/tmp")
        process = self._run_job_script(
            job_script_content, env={local.LocalJobScriptBuilder.ARRAY_TASK_ID: "1"}
        )
        self.assertEqual(process.stdout.decode("utf-8").strip(), "FOO_0")


class GridEngineExecutionTest(parameterized.TestCase):
    @parameterized.parameters(
        (executors.GridEngine(requirements=JobRequirements(gpu=1)), True),
        (executors.GridEngine(parallel_environments={"gpu": 1}), True),
        (executors.GridEngine(), False),
    )
    def test_sge_infer_gpu_request(self, executor, expected):
        use_gpu = gridengine.GridEngineJobScriptBuilder._is_gpu_requested(executor)
        self.assertEqual(use_gpu, expected)

    def test_gridengine_header(self):
        executor = executors.GridEngine(
            resources={"h_vmem": "1G"},
            parallel_environments={"gpu": 1},
            project="test",
            account="alloc",
            walltime=10 * xm.Min,
            extra_directives=["-test_extra 1"],
            queue="test_queue",
        )
        header = gridengine.header_from_executor("test_job", executor, None, "/logs")
        self.assertIn("#$ -l h_vmem=1G", header)
        self.assertIn("#$ -l h_rt=00:10:00", header)
        self.assertIn("#$ -pe gpu 1", header)
        self.assertIn("#$ -P test", header)
        self.assertIn("#$ -A alloc", header)
        self.assertIn("#$ -q test_queue", header)
        self.assertIn("#$ -test_extra 1", header)

        executor = executors.GridEngine(max_parallel_tasks=2)
        header = gridengine.header_from_executor("test_job", executor, 10, "/logs")
        self.assertIn("#$ -t 1-10", header)
        self.assertIn("#$ -tc 2", header)

    def test_setup_cmds(self):
        executable = executables.AppBundle(
            name="test",
            entrypoint_command="echo hello",
            resource_uri="",
            container_image=_create_singularity_image("docker://python:3.10-slim"),
        )
        executor = executors.GridEngine(modules=["module1"])
        with patch.object(
            gridengine.GridEngineJobScriptBuilder,
            "_is_gpu_requested",
            return_value=True,
        ):
            setup_cmds = (
                gridengine.GridEngineJobScriptBuilder._create_job_script_prologue(
                    executable, executor
                )
            )
        self.assertIn("nvidia-smi", setup_cmds)
        self.assertIn("module load module1", setup_cmds)
        self.assertIn("singularity --version", setup_cmds)

    def test_launch(self):
        staging_dir = self.create_tempdir(name="staging")

        archive_name = "archive.zip"
        container_name = "container.sif"
        archive = staging_dir.create_file(archive_name)
        container = staging_dir.create_file(container_name)

        deploy_dir = self.create_tempdir(name="deploy")
        executable = executables.AppBundle(
            name="test",
            entrypoint_command="echo hello",
            resource_uri=archive.full_path,
            container_image=_create_singularity_image(container.full_path),
        )
        executor = executors.GridEngine()
        job = xm.Job(executable, executor, name="test")
        artifact = artifacts.ArtifactStore(
            fsspec.filesystem("local"), deploy_dir.full_path
        )
        settings = config.ClusterSettings({})
        client = gridengine.GridEngineClient(settings, artifact)
        with mock.patch.object(
            gridengine_cluster.GridEngineCluster, "launch"
        ) as mock_launch:
            mock_launch.return_value = "1"
            client.launch("test_job", job)


class SlurmExecutionTest(parameterized.TestCase):
    def test_slurm_header(self):
        executor = executors.Slurm(
            resources={"mem": "1G"},
            walltime=10 * xm.Hr,
            exclusive=True,
            partition="contrib-gpu-long",
            extra_directives=["--test_extra 1"],
        )
        header = slurm.header_from_executor("test_job", executor, None, "/logs")
        self.assertIn("#SBATCH --mem=1G", header)
        self.assertIn("#SBATCH --time=10:00:00", header)
        self.assertIn("#SBATCH --exclusive", header)
        self.assertIn("#SBATCH --partition=contrib-gpu-long", header)
        self.assertIn("#SBATCH --test_extra 1", header)

        executor = executors.Slurm()
        header = slurm.header_from_executor("test_job", executor, 10, "/logs")
        self.assertIn("#SBATCH --array=1-10", header)

    def test_setup_cmds(self):
        executable = executables.AppBundle(
            name="test",
            entrypoint_command="echo hello",
            resource_uri="",
            container_image=_create_singularity_image("docker://python:3.10-slim"),
        )
        executor = executors.Slurm(modules=["module1"])
        with patch.object(
            gridengine.GridEngineJobScriptBuilder,
            "_is_gpu_requested",
            return_value=True,
        ):
            setup_cmds = slurm.SlurmJobScriptBuilder._create_job_script_prologue(
                executable, executor
            )
        self.assertIn("module load module1", setup_cmds)

    def test_slurm_launch(self):
        staging_dir = self.create_tempdir(name="staging")

        archive_name = "archive.zip"
        container_name = "container.sif"
        archive = staging_dir.create_file(archive_name)
        container = staging_dir.create_file(container_name)

        deploy_dir = self.create_tempdir(name="deploy")
        executable = executables.AppBundle(
            name="test",
            entrypoint_command="echo hello",
            resource_uri=archive.full_path,
            container_image=_create_singularity_image(
                container.full_path,
            ),
        )
        executor = executors.Slurm()
        job = xm.Job(executable, executor, name="test")
        artifact = artifacts.ArtifactStore(
            fsspec.filesystem("local"), deploy_dir.full_path
        )
        settings = config.ClusterSettings({})
        client = slurm.SlurmClient(settings, artifact)
        with mock.patch.object(slurm_cluster.SlurmCluster, "launch") as mock_launch:
            mock_launch.return_value = slurm_cluster.parse_job_id("job 1")
            client.launch("test_job", job)


if __name__ == "__main__":
    absltest.main()
