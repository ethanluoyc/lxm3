import os
import subprocess
import tempfile


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
""".format(base_image=base_image, lock_file=lock_file)


def python_container_dockerfile(base_image: str, requirements: str):
    return """\
FROM {base_image}
RUN if ! id 1000; then useradd -m -u 1000 docker; fi

COPY {requirements} /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
""".format(base_image=base_image, requirements=requirements)


def build_image_by_dockerfile(image_name: str, dockerfile: str, path: str):
    subprocess.run(
        ["docker", "buildx", "build", "-t", image_name, "-f", dockerfile, path],
        check=True,
    )


def build_image_by_dockerfile_content(
    image_name: str, dockerfile_content: str, path: str
):
    with tempfile.TemporaryDirectory() as tmpdir:
        dockerfile = os.path.join(tmpdir, "Dockerfile")
        with open(dockerfile, "w") as f:
            f.write(dockerfile_content)
        build_image_by_dockerfile(image_name, dockerfile, path)
