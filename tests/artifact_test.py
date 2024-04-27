import os
import pathlib

import fsspec
from absl.testing import absltest
from absl.testing import parameterized

from lxm3.xm_cluster import artifacts


class LocalArtifactsTest(parameterized.TestCase):
    def _create_store(self):
        fs = fsspec.filesystem("local")
        staging = self.create_tempdir()
        project = "test"
        store = artifacts.ArtifactStore(fs, os.fspath(staging), project)
        return store, staging, project

    def test_put_text(self):
        store, staging, project = self._create_store()
        content = "foo"
        dst = "nested/one"
        expected_dst = pathlib.Path(staging, f"projects/{project}/{dst}")
        store.put_text(content, dst)
        self.assertEqual(content, expected_dst.read_text())
        self.assertEqual(store.get_file_info(dst).size, len(content))

    def test_put_file(self):
        store, staging, project = self._create_store()
        content = "foo"
        src = self.create_tempfile("test.py", content=content)
        dst = "nested/one"
        expected_dst = pathlib.Path(staging, f"projects/{project}/{dst}")
        store.put_file(src.full_path, dst)

        self.assertEqual(content, expected_dst.read_text())
        self.assertEqual(store.get_file_info(dst).size, len(content))

    def test_ensure_dir(self):
        store, staging, project = self._create_store()
        dst = "nested/one"
        store.ensure_dir(dst)
        expected_path = pathlib.Path(staging, f"projects/{project}", dst)
        self.assertTrue(expected_path.exists())
        self.assertTrue(expected_path.is_dir())


if __name__ == "__main__":
    absltest.main()
