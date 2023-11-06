import os
import unittest.mock
import zipfile

from absl.testing import absltest
from absl.testing import parameterized

from lxm3 import xm
from lxm3 import xm_cluster
from lxm3.xm_cluster import artifacts
from lxm3.xm_cluster.packaging import cluster

_HERE = os.path.abspath(os.path.dirname(__file__))


class PackagingTest(parameterized.TestCase):
    @parameterized.parameters(
        (cluster._package_python_package,),
    )
    def test_package_python(self, pkg_fun):
        spec = xm_cluster.PythonPackage(
            entrypoint=xm_cluster.ModuleName("py_package.main"),
            path=os.path.join(_HERE, "testdata/test_pkg"),
        )

        with unittest.mock.patch("subprocess.run"):
            tmpdir = self.create_tempdir().full_path
            store = artifacts.LocalArtifactStore(tmpdir, "test")
            executable = pkg_fun(
                spec,
                xm.Packageable(
                    spec,
                    xm_cluster.Local().Spec(),
                ),
                store,
            )
            self.assertIsInstance(executable, xm_cluster.Command)

    def test_package_default_pip_args(self):
        spec = xm_cluster.PythonPackage(
            entrypoint=xm_cluster.ModuleName("py_package.main"),
            path=os.path.join(_HERE, "testdata/test_pkg"),
        )
        self.assertEqual(
            sorted(spec.pip_args),
            sorted(["--no-deps", "--no-compile"]),
        )

    @parameterized.parameters(
        (cluster._package_universal_package,),
    )
    def test_package_universal(self, pkg_fun):
        spec = xm_cluster.UniversalPackage(
            entrypoint=["python3", "main.py"],
            path=os.path.join(_HERE, "testdata/test_universal"),
            build_script="build.sh",
        )
        tmpdir = self.create_tempdir().full_path
        store = artifacts.LocalArtifactStore(tmpdir, "test")
        executable = pkg_fun(
            spec,
            xm.Packageable(
                spec,
                xm_cluster.Local().Spec(),
            ),
            store,
        )
        self.assertIsInstance(executable, xm_cluster.Command)
        # Check that archive exists
        self.assertTrue(store._fs.exists(executable.resource_uri))
        with store._fs.open(executable.resource_uri, "rb") as f:
            archive = zipfile.ZipFile(f)  # type: ignore
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
