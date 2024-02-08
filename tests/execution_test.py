import os
import shutil
import subprocess
import sys
import unittest
import zipfile
from unittest import mock
from unittest.mock import patch

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


def is_singularity_installed():
    return shutil.which("singularity") is not None


def is_docker_installed():
    return shutil.which("docker") is not None


class JobScriptBuilderTest(parameterized.TestCase):
    def test_env_vars(self):
        env_var_str = job_script.JobScriptBuilder._create_env_vars(
            [{"FOO": "BAR1"}, {"FOO": "BAR2"}]
        )
        expected = """\
FOO_0="BAR1"
FOO_1="BAR2"
FOO=$(eval echo \\$"FOO_$LXM_TASK_ID")
export FOO"""
        self.assertEqual(env_var_str, expected)

    def test_empty_env_vars(self):
        self.assertEqual(job_script.JobScriptBuilder._create_env_vars([{}]), "")

    def test_common_values(self):
        env_var_str = job_script.JobScriptBuilder._create_env_vars(
            [{"FOO": "BAR", "BAR": "1"}, {"FOO": "BAR", "BAR": "2"}]
        )
        expected = """\
export FOO="BAR"
BAR_0="1"
BAR_1="2"
BAR=$(eval echo \\$"BAR_$LXM_TASK_ID")
export BAR"""
        self.assertEqual(env_var_str, expected)

    def test_different_keys(self):
        with self.assertRaises(ValueError):
            job_script.JobScriptBuilder._create_env_vars(
                [{"FOO": "BAR1"}, {"BAR": "BAR2"}]
            )

    def test_args(self):
        args_str = job_script.JobScriptBuilder._create_args(
            [["--seed=1", "--task=1"], ["--seed=2", "--task=2"]]
        )
        expected = """\
TASK_CMD_ARGS_0="--seed=1 --task=1"
TASK_CMD_ARGS_1="--seed=2 --task=2"
TASK_CMD_ARGS=$(eval echo \\$"TASK_CMD_ARGS_$LXM_TASK_ID")
eval set -- $TASK_CMD_ARGS"""
        self.assertEqual(args_str, expected)

    def test_empty_args(self):
        self.assertEqual(job_script.JobScriptBuilder._create_args([]), "")
        self.assertEqual(
            job_script.JobScriptBuilder._create_args([[]]),
            """\
TASK_CMD_ARGS_0=""
TASK_CMD_ARGS=$(eval echo \\$"TASK_CMD_ARGS_$LXM_TASK_ID")
eval set -- $TASK_CMD_ARGS""",
        )

    def test_get_additional_env(self):
        job_env = {"FOO": "FOO_0", "OVERRIDE": "OVERRIDE"}

        self.assertEqual(
            job_script.JobScriptBuilder._get_additional_env(
                job_env, {"FOO": "FOO_HOST", "BAR": "BAR"}
            ),
            {"BAR": "BAR"},
        )

    def test_additional_bindings(self):
        job_binds = {"/c": "/b", "/d": "/e"}
        overrides = {"/a": "/b", "/foo": "/bar"}
        additional_binds = job_script.JobScriptBuilder._get_additional_binds(
            job_binds, overrides
        )
        self.assertEqual(additional_binds, {"/foo": "/bar"})


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
        executable = executables.Command(
            name="test",
            entrypoint_command="echo hello",
            resource_uri=archive.full_path,
            singularity_image=container.full_path,
        )
        executor = executors.Local()
        job = xm.Job(executable, executor, name="test")
        artifact = artifacts.LocalArtifactStore(deploy_dir.full_path)
        settings = config.LocalSettings()
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

    def test_job_script_run_single_job(self):
        tmpf = self.create_tempfile("test.zip")
        with zipfile.ZipFile(tmpf.full_path, "w") as z:
            info = zipfile.ZipInfo("entrypoint.sh")
            info.external_attr = 0o777 << 16  # give full access to included file
            z.writestr(
                info,
                """\
#!/usr/bin/env bash
echo $@ $FOO""",
            )
        executable = xm_cluster.Command(
            "foo", "./entrypoint.sh", resource_uri=tmpf.full_path
        )
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
        tmpf = self.create_tempfile("test.zip")
        with zipfile.ZipFile(tmpf.full_path, "w") as z:
            info = zipfile.ZipInfo("entrypoint.sh")
            info.external_attr = 0o777 << 16  # give full access to included file
            z.writestr(
                info,
                """\
#!/usr/bin/env bash
echo $@ $FOO""",
            )
        executable = xm_cluster.Command(
            "foo", "./entrypoint.sh", resource_uri=tmpf.full_path
        )
        job = xm_cluster.ArrayJob(
            executable,
            executor=xm_cluster.Local(),
            args=[{"seed": 1}, {"seed": 2}],
            env_vars=[{"FOO": "FOO_0"}, {"FOO": "FOO_1"}],
        )
        builder = local.LocalJobScriptBuilder()
        job_script_content = builder.build(job, "foo", "/tmp")
        process = self._run_job_script(
            job_script_content, env={builder.TASK_ID_VAR_NAME: "1"}
        )
        self.assertEqual(process.stdout.decode("utf-8").strip(), "--seed=1 FOO_0")
        process = self._run_job_script(
            job_script_content, env={builder.TASK_ID_VAR_NAME: "2"}
        )
        self.assertEqual(process.stdout.decode("utf-8").strip(), "--seed=2 FOO_1")

    def test_job_script_handles_ml_collections_quoting(self):
        tmpf = self.create_tempfile("test.zip")
        with zipfile.ZipFile(tmpf.full_path, "w") as z:
            info = zipfile.ZipInfo("entrypoint.sh")
            info.external_attr = 0o777 << 16  # give full access to included file
            z.writestr(
                "run.py",
                """\
from absl import app
from ml_collections import config_dict
from ml_collections import config_flags

def _get_config():
    config = config_dict.ConfigDict()
    config.name = ""
    return config


_CONFIG = config_flags.DEFINE_config_dict("config", _get_config())


def main(_):
    config = _CONFIG.value
    print(config.name)

if __name__ == "__main__":
    config = _get_config()
    app.run(main)

""",
            )
            z.writestr(
                info,
                """\
#!/usr/bin/env bash
{} run.py $@
""".format(
                    sys.executable
                ),
            )
        executable = xm_cluster.Command(
            "foo", "./entrypoint.sh", resource_uri=tmpf.full_path
        )
        job = xm_cluster.ArrayJob(
            executable,
            executor=xm_cluster.Local(),
            args=[{"config.name": "train[:90%]"}],
            env_vars=[{"FOO": "FOO_0"}],
        )
        builder = local.LocalJobScriptBuilder()
        job_script_content = builder.build(job, "foo", "/tmp")
        process = self._run_job_script(job_script_content, env={"SGE_TASK_ID": "1"})
        self.assertEqual(process.stdout.decode("utf-8").strip(), "train[:90%]")

    @unittest.skipIf(not is_singularity_installed(), "Singularity is not installed")
    def test_singularity(self):
        tmpf = self.create_tempfile("test.zip")
        with zipfile.ZipFile(tmpf.full_path, "w") as z:
            info = zipfile.ZipInfo("entrypoint.sh")
            info.external_attr = 0o777 << 16  # give full access to included file
            z.writestr(
                info,
                """\
#!/usr/bin/env bash
echo $FOO""",
            )

        executable = xm_cluster.Command(
            "test",
            "./entrypoint.sh",
            resource_uri=tmpf.full_path,
            singularity_image="docker://python:3.10-slim",
        )
        job = xm_cluster.ArrayJob(
            executable, executor=xm_cluster.Local(), env_vars=[{"FOO": "FOO_0"}]
        )
        job_script_content = local.LocalJobScriptBuilder().build(job, "foo", "/tmp")
        process = self._run_job_script(
            job_script_content, env={local.LocalJobScriptBuilder.TASK_ID_VAR_NAME: "1"}
        )
        self.assertEqual(process.stdout.decode("utf-8").strip(), "FOO_0")

    @unittest.skipIf(not is_docker_installed(), "Docker is not installed")
    def test_docker_image(self):
        tmpf = self.create_tempfile("test.zip")
        with zipfile.ZipFile(tmpf.full_path, "w") as z:
            info = zipfile.ZipInfo("entrypoint.sh")
            info.external_attr = 0o777 << 16  # give full access to included file
            z.writestr(
                info,
                """\
#!/usr/bin/env bash
echo $FOO""",
            )

        executable = xm_cluster.Command(
            "test",
            "./entrypoint.sh",
            resource_uri=tmpf.full_path,
            docker_image="python:3.10-slim",
        )
        job = xm_cluster.ArrayJob(
            executable, executor=xm_cluster.Local(), env_vars=[{"FOO": "FOO_0"}]
        )
        job_script_content = local.LocalJobScriptBuilder().build(job, "foo", "/tmp")
        process = self._run_job_script(
            job_script_content, env={local.LocalJobScriptBuilder.TASK_ID_VAR_NAME: "1"}
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
        executable = executables.Command(
            name="test",
            entrypoint_command="echo hello",
            resource_uri="",
            singularity_image="docker://python:3.10-slim",
        )
        executor = executors.GridEngine(modules=["module1"])
        with patch.object(
            gridengine.GridEngineJobScriptBuilder,
            "_is_gpu_requested",
            return_value=True,
        ):
            setup_cmds = gridengine.GridEngineJobScriptBuilder._create_setup_cmds(
                executable, executor
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
        executable = executables.Command(
            name="test",
            entrypoint_command="echo hello",
            resource_uri=archive.full_path,
            singularity_image=container.full_path,
        )
        executor = executors.GridEngine()
        job = xm.Job(executable, executor, name="test")
        artifact = artifacts.LocalArtifactStore(deploy_dir.full_path)
        settings = config.ClusterSettings()
        client = gridengine.GridEngineClient(settings, artifact)
        with mock.patch.object(
            gridengine_cluster.GridEngineCluster, "launch"
        ) as mock_launch:
            mock_launch.return_value = gridengine_cluster.parse_job_id("1")
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
        executable = executables.Command(
            name="test",
            entrypoint_command="echo hello",
            resource_uri="",
            singularity_image="docker://python:3.10-slim",
        )
        executor = executors.Slurm(modules=["module1"])
        with patch.object(
            gridengine.GridEngineJobScriptBuilder,
            "_is_gpu_requested",
            return_value=True,
        ):
            setup_cmds = slurm.SlurmJobScriptBuilder._create_setup_cmds(
                executable, executor
            )
        self.assertIn("module load module1", setup_cmds)
        self.assertIn("singularity --version", setup_cmds)

    def test_slurm_launch(self):
        staging_dir = self.create_tempdir(name="staging")

        archive_name = "archive.zip"
        container_name = "container.sif"
        archive = staging_dir.create_file(archive_name)
        container = staging_dir.create_file(container_name)

        deploy_dir = self.create_tempdir(name="deploy")
        executable = executables.Command(
            name="test",
            entrypoint_command="echo hello",
            resource_uri=archive.full_path,
            singularity_image=container.full_path,
        )
        executor = executors.Slurm()
        job = xm.Job(executable, executor, name="test")
        artifact = artifacts.LocalArtifactStore(deploy_dir.full_path)
        settings = config.ClusterSettings()
        client = slurm.SlurmClient(settings, artifact)
        with mock.patch.object(slurm_cluster.SlurmCluster, "launch") as mock_launch:
            mock_launch.return_value = slurm_cluster.parse_job_id("job 1")
            client.launch("test_job", job)


if __name__ == "__main__":
    absltest.main()
