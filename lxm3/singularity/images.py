import os
import pathlib
import subprocess

import appdirs

from lxm3.singularity import images
from lxm3.singularity import uri
from lxm3.xm_cluster.console import console


def build_singularity_image(
    image_path: os.PathLike, build_spec: str, force: bool = True
):
    build_cmd = ["singularity", "build"]
    if force:
        build_cmd.append("--force")
    pathlib.Path(image_path).parent.mkdir(exist_ok=True, parents=True)
    cmd = build_cmd + [str(image_path), build_spec]
    subprocess.run(cmd, check=True)


def build_singularity_image_from_docker_daemon(singularity_image: str) -> str:
    transport, ref = uri.split(singularity_image)
    if transport != "docker-daemon":
        raise ValueError(
            f"Expected docker-daemon transport, got {transport} for {singularity_image}"
        )

    filename = uri.filename(singularity_image, "sif")
    build_cache_dir = pathlib.Path(appdirs.user_cache_dir("lxm3"), "singularity")
    cache_image_path = build_cache_dir / filename
    marker_file = cache_image_path.with_suffix(cache_image_path.suffix + ".image_id")
    try:
        import docker
    except ImportError:
        raise ValueError(
            "docker-py library is required to build Singularity images from docker-daemon"
        )

    client = docker.from_env()
    image_id: str = client.images.get(ref).id  # type: ignore

    should_rebuild = True
    if cache_image_path.exists():
        if marker_file.exists():
            old_image_id = pathlib.Path(marker_file).read_text().strip()
            should_rebuild = old_image_id != image_id
            if should_rebuild:
                console.log("Image ID changed, rebuilding...")
            else:
                console.log("Reusing cached image from", cache_image_path)
        else:
            should_rebuild = True
            console.log("Marker file does not exist, rebuilding...")
    else:
        should_rebuild = True
        console.log("Container cache does not exist, rebuilding...")

    if should_rebuild:
        cache_image_path.parent.mkdir(parents=True, exist_ok=True)
        images.build_singularity_image(cache_image_path, singularity_image, force=True)
        console.log("Cached image at", cache_image_path)
        pathlib.Path(marker_file).write_text(image_id)

    return str(cache_image_path)
