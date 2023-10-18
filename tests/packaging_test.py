import os
import unittest.mock
import zipfile

from absl.testing import absltest
from absl.testing import parameterized

from lxm3 import xm
from lxm3 import xm_cluster
from lxm3.xm_cluster import packaging

_HERE = os.path.abspath(os.path.dirname(__file__))


class PackagingTest(parameterized.TestCase):
    @unittest.mock.patch("subprocess.run")
    def test_package_python(self, mock_run):
        spec = xm_cluster.PythonPackage(
            entrypoint=xm_cluster.ModuleName("py_package.main"),
            path=os.path.join(_HERE, "testdata/test_pkg"),
        )
        executable = packaging._package_python_package(
            spec,
            xm.Packageable(
                spec,
                xm_cluster.Local().Spec(),
            ),
        )
        self.assertIsInstance(executable, xm_cluster.Command)

    def test_package_universal(self):
        spec = xm_cluster.UniversalPackage(
            entrypoint=["python3", "main.py"],
            path=os.path.join(_HERE, "testdata/test_universal"),
            build_script="build.sh",
        )
        executable = packaging._package_universal_package(
            spec,
            xm.Packageable(
                spec,
                xm_cluster.Local().Spec(),
            ),
        )
        self.assertIsInstance(executable, xm_cluster.Command)
        # Check that archive exists
        self.assertTrue(os.path.exists(executable.resource_uri))
        archive = zipfile.ZipFile(executable.resource_uri)
        self.assertEqual(set(archive.namelist()), set(["main.py"]))

    def test_package_universal_not_executable(self):
        with self.assertRaises(ValueError):
            xm_cluster.UniversalPackage(
                entrypoint=["python3", "main.py"],
                path=os.path.join(_HERE, "testdata/test_universal"),
                build_script="build_not_executable.sh",
            )

    def test_package_singularity_invalid_path(self):
        py_package = xm_cluster.PythonPackage(
            entrypoint=xm_cluster.ModuleName("py_package.main"),
            path=os.path.join(_HERE, "testdata/test_pkg"),
        )
        fake_image_path = self.create_tempfile().full_path
        # This is OK
        xm_cluster.SingularityContainer(py_package, image_path=fake_image_path)
        # Raises on non-existent path
        with self.assertRaises(ValueError):
            xm_cluster.SingularityContainer(
                py_package, image_path=os.path.join(_HERE, "/fake/image.sif")
            )


if __name__ == "__main__":
    absltest.main()
