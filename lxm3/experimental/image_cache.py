"""A simple cache for singularity images bootstraped from Docker/Podman.

This module provides the facility to cache singularity images that
are bootstrapped from a locally built docker/podman container image.
While Docker/Podman has a build cache that can avoids building
a new image if the dependencies have not changed, bootstrapping
from images cached by Docker/Podman still requires rebuilding
a new SIF image, which can be prohibitively slow for large images.
This module provides the utility to cache the SIF images built and
reuse the same SIF file if the image from the Docker/Podman daemon has
not changed.

It maintains singularity images as opaque blobs in
a managed directory. Each blob is named by the image id, instead of
its checksum so it's different from the OCI-layout.

In addition, it maintains a list of symlinks
to the blobs to identify each image tag with the corresponding SIF file.

For now, blobs are never deleted, and the cache is never pruned. When
a new image with the same tag is built, the symlink will be updated to
point to the new blob. Note that blobs may be shared between different
tags, so that the same underlying image is not duplicated in the cache.

TODOs:
    * Separate the cache to have separate directory per image to avoid
      name collision.
    * Implement garbage collection for blobs that are unreferenced.
    * Support keeping a cache size upperbound.
    * Implement logic for caching other transports (docker for example)
      so that users will never have to worry about manually maintaining
      a folder of images.
"""

import dataclasses
import os
import pathlib
import shutil
import subprocess
import tempfile

from lxm3 import singularity


@dataclasses.dataclass(frozen=True)
class ImageInfo:
    """A dataclass to represent the cached image."""

    # Digest of the image (source Image ID)
    digest: str
    # Path to the symlink (which includes the image name and sif suffix)
    path: str
    # Path to the actual image (with an opaque name, which is the digest of the original image)
    blob_path: str


class ImageCache:
    def __init__(self, cache_dir: str):
        self._cache_dir = pathlib.Path(cache_dir)
        self._blobs_dir = self._cache_dir / "blobs"
        self._blobs_dir.mkdir(parents=True, exist_ok=True)

    def image_path(self, name: str) -> pathlib.Path:
        return (self._cache_dir / name).absolute()

    def image_exists(self, name: str) -> bool:
        return self.image_path(name).exists()

    def get_image(self, name: str) -> ImageInfo:
        realpath = os.path.realpath(self.image_path(name))
        digest = os.path.basename(realpath)
        assert self.image_exists(name)
        return ImageInfo(
            digest=digest,
            path=str(self.image_path(name).absolute()),
            blob_path=realpath,
        )

    def blob_path(self, digest: str) -> pathlib.Path:
        return (self._blobs_dir / digest).absolute()

    def blob_exists(self, digest: str) -> bool:
        return self.blob_path(digest).exists()

    def put_blob(self, digest: str, path: os.PathLike):
        shutil.copy2(path, str(self.blob_path(digest)) + ".tmp")
        os.rename(str(self.blob_path(digest)) + ".tmp", self.blob_path(digest))

    def link(self, name: str, digest: str):
        assert self.blob_exists(digest)
        if self.image_path(name).exists():
            os.remove(self.image_path(name))
        os.symlink(
            os.path.relpath(self.blob_path(digest), self._cache_dir),
            self.image_path(name),
        )


def get_cached_image(image_spec: str, cache_dir: str) -> ImageInfo:
    """Get the cached image if it exists, otherwise build and cache it.
    Args:
        image_spec: The image specification to be bootstrapped.
            It currently only supports docker-daemon and podman-daemon.
        cache_dir: The directory to store the cache.
            For each transport (docker-daemon, podman-daemon), a separate
            subdirectory is created to store the cache.

    Returns:
        An `ImageInfo` identified the cached image.

    Examples:
        # Assuming cache is first empty
        # Will build the image and cache it
        info = get_cached_image("docker-daemon://python:3.10-slim", "/tmp/image_cache")

        # Running again will reuse the cached image
        info = get_cached_image("docker-daemon://python:3.10-slim", "/tmp/image_cache")

    """
    transport, ref = singularity.uri.split(image_spec)
    image_name = singularity.uri.filename(image_spec, "sif")

    if transport == "docker-daemon":
        try:
            import docker
        except ImportError:
            raise ImportError(
                "docker package is required to use docker daemon transport"
            )
        cache = ImageCache(os.path.join(cache_dir, "docker-daemon"))
        client = docker.from_env()
        digest = client.images.get(ref).id
        digest = str(digest)
        del client

    elif transport == "podman-daemon":
        try:
            import podman
        except ImportError:
            raise ImportError(
                "podman package is required to use docker daemon transport"
            )
        cache = ImageCache(os.path.join(cache_dir, "podman-daemon"))
        client = podman.PodmanClient()
        if ref.startswith("//"):
            ref = ref[2:]
        digest = str(client.images.get(ref).id)
        del client
    else:
        raise ValueError(f"Unsupported transport {transport}")

    if cache.image_exists(image_name):
        image_info = cache.get_image(image_name)
        assert image_info is not None
        if image_info.digest != digest:
            if cache.blob_exists(digest):
                print(f"Reusing cached blob {digest} for image {image_name}")
                # Image does not exist but blob does
                # Just link
                cache.link(image_name, digest)
            else:
                print(f"Rebuilding image {image_name} from {image_spec}")
                with tempfile.TemporaryDirectory() as tmpdir:
                    build_image_path = pathlib.Path(tmpdir) / image_name
                    if transport == "docker-daemon":
                        singularity.images.build_singularity_image(
                            build_image_path, image_spec
                        )
                    else:
                        subprocess.run(
                            [
                                "podman",
                                "save",
                                "--format=oci-archive",
                                "-o",
                                os.path.join(tmpdir, "image.tar"),
                                ref,
                            ],
                            check=True,
                        )
                        singularity.images.build_singularity_image(
                            build_image_path,
                            f"oci-archive://{os.path.join(tmpdir, 'image.tar')}",
                        )

                    # First put blob
                    cache.put_blob(digest, build_image_path)
                    # Update image links
                    cache.link(image_name, digest)
        else:
            print(f"Reusing cached image {image_info.path} from {digest}")
        return cache.get_image(image_name)
    else:
        if cache.blob_exists(digest):
            print(f"Link image {image_name} to {cache.blob_path(digest)}")
            # Image does not exist but blob does
            # Just link
            cache.link(image_name, digest)
        else:
            # Neither image nor blob exists
            print(f"Rebuilding image {image_name} from {image_spec}")
            with tempfile.TemporaryDirectory() as tmpdir:
                build_image_path = pathlib.Path(tmpdir) / image_name
                if transport == "docker-daemon":
                    singularity.images.build_singularity_image(
                        build_image_path, image_spec
                    )
                else:
                    subprocess.run(
                        [
                            "podman",
                            "save",
                            "--format=oci-archive",
                            "-o",
                            os.path.join(tmpdir, "image.tar"),
                            ref,
                        ],
                        check=True,
                    )
                    singularity.images.build_singularity_image(
                        build_image_path,
                        f"oci-archive://{os.path.join(tmpdir, 'image.tar')}",
                    )
                # First put blob
                cache.put_blob(digest, build_image_path)
                # Update image links
                cache.link(image_name, digest)

        return cache.get_image(image_name)


__all__ = ["get_cached_image", "ImageInfo"]
