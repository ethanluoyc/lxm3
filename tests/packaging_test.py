import os
import sys
import unittest
import unittest.mock
import zipfile

import fsspec
from absl.testing import absltest
from absl.testing import parameterized

from lxm3 import xm
from lxm3 import xm_cluster
from lxm3.singularity import image_cache
from lxm3.xm_cluster import artifacts
from lxm3.xm_cluster.packaging import router

_HERE = os.path.abspath(os.path.dirname(__file__))


def _create_artifact_store(staging, project):
    return artifacts.ArtifactStore(fsspec.filesystem("local"), staging, project)


class PackagingTest(parameterized.TestCase):
    def _create_test_artifact_store(self):
        tmpdir = self.create_tempdir().full_path
        return _create_artifact_store(tmpdir, "test")

    def test_package_python(self):
        spec = xm_cluster.PythonPackage(
            entrypoint=xm_cluster.ModuleName("py_package.main"),
            path=os.path.join(_HERE, "testdata/test_pkg"),
        )

        with (
            unittest.mock.patch.object(
                router,
                "_get_artifact_store",
                return_value=self._create_test_artifact_store(),
            ),
            unittest.mock.patch("subprocess.run"),
        ):
            executable = router.packaging_router(
                xm.Packageable(spec, xm_cluster.Local().Spec())
            )
            self.assertIsInstance(executable, xm_cluster.AppBundle)

    def test_package_default_pip_args(self):
        spec = xm_cluster.PythonPackage(
            entrypoint=xm_cluster.ModuleName("py_package.main"),
            path=os.path.join(_HERE, "testdata/test_pkg"),
        )
        self.assertEqual(
            sorted(spec.pip_args),
            sorted(["--no-deps", "--no-compile"]),
        )

    @absltest.skipIf("darwin" in sys.platform, "Not working on MacOS")
    def test_package_universal(self):
        spec = xm_cluster.UniversalPackage(
            entrypoint=["python3", "main.py"],
            path=os.path.join(_HERE, "testdata/test_universal"),
            build_script="build.sh",
        )
        store = self._create_test_artifact_store()
        with unittest.mock.patch.object(
            router, "_get_artifact_store", return_value=store
        ):
            executable = router.packaging_router(
                xm.Packageable(spec, xm_cluster.Local().Spec())
            )
            self.assertIsInstance(executable, xm_cluster.AppBundle)
            # Check that archive exists
            self.assertTrue(store.filesystem.exists(executable.resource_uri))
            with store.filesystem.open(executable.resource_uri, "rb") as f:
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

    def test_push_singularity_image_docker_daemon(self):
        store = self._create_test_artifact_store()
        image_name = "docker-daemon://python:3.10-slim"
        with unittest.mock.patch.object(
            image_cache, "get_cached_image"
        ) as mock_get_cached_image:
            blob = self.create_tempfile()
            mock_get_cached_image.return_value = image_cache.ImageInfo(
                digest="12345",
                path=blob.full_path,
                blob_path=blob.full_path,
            )
            router._maybe_push_singularity_image(image_name, artifact_store=store)
            mock_get_cached_image.assert_called_once()

    def test_package_pex(self):
        spec = xm_cluster.PexBinary(
            entrypoint=xm_cluster.ModuleName("py_package.main"),
            packages=["py_package"],
            path=os.path.join(_HERE, "testdata/test_pkg"),
        )

        with (
            unittest.mock.patch.object(
                router,
                "_get_artifact_store",
                return_value=self._create_test_artifact_store(),
            ),
            unittest.mock.patch("subprocess.run"),
        ):
            executable = router.packaging_router(
                xm.Packageable(spec, xm_cluster.Local().Spec())
            )
            self.assertIsInstance(executable, xm_cluster.AppBundle)


if __name__ == "__main__":
    absltest.main()
