import os
import pathlib
import subprocess


def build_singularity_image(
    image_path: os.PathLike, build_spec: str, force: bool = True
):
    build_cmd = ["singularity", "build"]
    if force:
        build_cmd.append("--force")
    pathlib.Path(image_path).parent.mkdir(exist_ok=True, parents=True)
    cmd = build_cmd + [str(image_path), build_spec]
    subprocess.run(cmd, check=True)
