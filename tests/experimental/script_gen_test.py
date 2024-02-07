import os
import shlex
import shutil
import subprocess
import unittest

from absl.testing import absltest
from absl.testing import parameterized

from lxm3.experimental import job_script_builder

HERE = os.path.dirname(__file__)


singularity_image = "examples/basic/python_3.10-slim.sif"


@unittest.skip("Skip until we have a better way to test this")
class ScriptGenTest(parameterized.TestCase):
    @parameterized.parameters((False,), (True,))
    def test_array_script(self, use_singularity):
        tmpdir = self.create_tempdir()
        f = shutil.make_archive(
            os.path.join(tmpdir, "test"), "zip", os.path.join(HERE, "testdata")
        )
        per_task_envs = [{"BAR": "0"}, {"BAR": "1"}]
        per_task_args = [
            ["--foo", "1", "--notes", shlex.quote("space space")],
            ["--foo", "2", "--notes", shlex.quote("space space2")],
        ]

        script = job_script_builder.create_job_script(
            command=["./pkg/main.py"],
            archives=f,
            singularity_image=singularity_image if use_singularity else None,  # type: ignore
            singularity_options=["--compat"],
            per_task_args=per_task_args,
            per_task_envs=per_task_envs,
        )
        job_script = tmpdir.create_file("jobscript.sh", content=script)

        proc = subprocess.run(
            ["sh", job_script.full_path],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "SGE_TASK_ID": "2"},
        )
        self.assertIn("['--foo', '2', '--notes', 'space space2']", proc.stdout)
        with self.assertRaises(subprocess.CalledProcessError):
            print(
                subprocess.run(
                    ["sh", job_script.full_path],
                    check=True,
                    capture_output=True,
                    text=True,
                    env={**os.environ},
                ).stderr
            )

    @parameterized.parameters((True,), (False,))
    def test_job_script(self, use_singularity):
        tmpdir = self.create_tempdir()
        f = shutil.make_archive(
            os.path.join(tmpdir, "test"), "zip", os.path.join(HERE, "testdata")
        )

        script = job_script_builder.create_job_script(
            command=["pkg/main.py", "arg0", "arg1"],
            archives=f,
            singularity_options=["--compat"],
            singularity_image=singularity_image if use_singularity else None,  # type: ignore
        )

        job_script = tmpdir.create_file("jobscript.sh", content=script)

        proc = subprocess.run(
            ["sh", job_script.full_path], capture_output=True, check=True, text=True
        )
        self.assertIn("['arg0', 'arg1']", proc.stdout)


if __name__ == "__main__":
    absltest.main()
