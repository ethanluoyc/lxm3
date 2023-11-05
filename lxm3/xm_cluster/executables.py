from typing import Dict, Optional

import attr

from lxm3 import xm


@attr.s(auto_attribs=True)
class Command(xm.Executable):
    entrypoint_command: str
    resource_uri: str
    args: xm.SequentialArgs = attr.Factory(xm.SequentialArgs)
    env_vars: Dict[str, str] = attr.Factory(dict)
    singularity_image: Optional[str] = None
    docker_image: Optional[str] = None
