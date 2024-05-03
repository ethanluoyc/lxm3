import shutil


def is_singularity_installed():
    return shutil.which("singularity") is not None


def is_docker_installed():
    return shutil.which("docker") is not None
