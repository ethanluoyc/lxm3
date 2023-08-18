import os
import subprocess
from typing import List, Optional, Sequence


def rsync(
    src: str,
    dst: str,
    opt: List[str],
    host: Optional[str] = None,
    excludes: Optional[Sequence[str]] = None,
    filters: Optional[Sequence[str]] = None,
    mkdirs: bool = False,
):
    if excludes is None:
        excludes = []
    if filters is None:
        filters = []
    opt = list(opt)
    for exclude in excludes:
        opt.append(f"--exclude={exclude}")
    for filter in filters:
        opt.append(f"--filter=:- {filter}")
    if not host:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        sync_cmd = ["rsync"] + opt + [src, dst]
        subprocess.check_call(sync_cmd)
    else:
        if mkdirs:
            subprocess.check_output(["ssh", host, "mkdir", "-p", os.path.dirname(dst)])
        dst = f"{host}:{dst}"
        sync_cmd = ["rsync"] + opt + [src, dst]
        subprocess.check_call(sync_cmd)
