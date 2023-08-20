import os
import unittest.mock

from absl.testing import absltest
from absl.testing import parameterized

from lxm3 import xm
from lxm3 import xm_cluster
from lxm3.xm_cluster import packaging


class PackagingTest(parameterized.TestCase):
    @unittest.mock.patch("subprocess.run")
    def test_package_python(self, mock_run):
        spec = xm_cluster.PythonPackage(
            entrypoint=xm_cluster.ModuleName("py_package.main"),
            path=os.path.join(os.path.dirname(__file__), "testdata/test_pkg"),
        )
        executable = packaging._package_python_package(
            spec,
            xm.Packageable(
                spec,
                xm_cluster.Local().Spec(),
            ),
        )
        self.assertIsInstance(executable, xm_cluster.Command)


if __name__ == "__main__":
    absltest.main()
