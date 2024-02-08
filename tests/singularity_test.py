# https://github.com/sylabs/singularity/blob/main/internal/pkg/util/uri/uri_test.go
from absl.testing import absltest
from absl.testing import parameterized

from lxm3 import singularity


class URITest(parameterized.TestCase):
    # fmt: off
    @parameterized.named_parameters(
        ("docker basic", "docker://ubuntu", "sif", "ubuntu_latest.sif"),
        ("docker scoped", "docker://user/image", "oci.sif", "image_latest.oci.sif"),
        ("dave's magical lolcow", "docker://sylabs.io/lolcow", "sif", "lolcow_latest.sif"),
        ("docker w tags", "docker://sylabs.io/lolcow:3.7", "sif", "lolcow_3.7.sif"),
    )
    # fmt: on
    def test_get_name(self, uri, suffix, expected):
        n = singularity.uri.filename(uri, suffix)
        self.assertEqual(
            n,
            expected,
            msg='incorrectly parsed name as "%s" (expected "%s")' % (n, expected),
        )

    # fmt: off
    @parameterized.named_parameters(
        ("docker basic", "docker://ubuntu", "docker", "//ubuntu"),
        ("docker scoped", "docker://user/image", "docker", "//user/image"),
        ("dave's magical lolcow", "docker://sylabs.io/lolcow", "docker", "//sylabs.io/lolcow"),
        ("docker with tags", "docker://sylabs.io/lolcow:latest", "docker", "//sylabs.io/lolcow:latest"),
        ["library basic", "library://image", "library", "//image"],
        ("library scoped", "library://collection/image", "library", "//collection/image"),
        ("without transport", "ubuntu", "", "ubuntu"),
        ("without transport with colon", "ubuntu:18.04.img", "", "ubuntu:18.04.img"),
    )
    # fmt: on
    def test_split(self, uri, transport, ref):
        tr, r = singularity.uri.split(uri)
        self.assertEqual(
            (tr, r),
            (transport, ref),
            msg="incorrectly parsed uri as %s : %s (expected %s : %s)"
            % (tr, r, transport, ref),
        )

    def test_build_image_from_daemon(self):
        build_spec = "docker-daemon://jax-cuda:latest"
        transport, ref = singularity.uri.split(build_spec)
        self.assertEqual(transport, "docker-daemon")
        filename = singularity.uri.filename(build_spec, "sif")
        self.assertEqual(filename, "jax-cuda_latest.sif")


if __name__ == "__main__":
    absltest.main()
