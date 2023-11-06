import os
import pathlib
import subprocess
import tempfile

import appdirs

from lxm3 import singularity
from lxm3.xm_cluster.console import console


def pdm_dockerfile(base_image: str, lock_file: str):
    return """\
FROM {base_image} as builder
RUN if ! id 1000; then useradd -m -u 1000 docker; fi
RUN pip install pdm

COPY {lock_file} /app/pdm.lock
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md

WORKDIR /app
RUN pdm install && pdm export > /requirements.txt

FROM {base_image}
COPY --from=builder /requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
""".format(
        base_image=base_image, lock_file=lock_file
    )


def python_container_dockerfile(base_image: str, requirements: str):
    return """\
FROM {base_image}
RUN if ! id 1000; then useradd -m -u 1000 docker; fi

COPY {requirements} /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
""".format(
        base_image=base_image, requirements=requirements
    )


def build_docker_image(image_name: str, dockerfile_content: str, path: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        dockerfile = os.path.join(tmpdir, "Dockerfile")
        with open(dockerfile, "w") as f:
            f.write(dockerfile_content)
        subprocess.run(
            ["docker", "buildx", "build", "-t", image_name, "-f", dockerfile, path],
            check=True,
        )


def build_singularity_image_from_docker_daemon(singularity_image: str) -> str:
    transport, ref = singularity.uri.split(singularity_image)
    if transport != "docker-daemon":
        raise ValueError(
            f"Expected docker-daemon transport, got {transport} for {singularity_image}"
        )

    filename = singularity.uri.filename(singularity_image, "sif")
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
        singularity.images.build_singularity_image(
            cache_image_path, singularity_image, force=True
        )
        console.log("Cached image at", cache_image_path)
        pathlib.Path(marker_file).write_text(image_id)

    return str(cache_image_path)
