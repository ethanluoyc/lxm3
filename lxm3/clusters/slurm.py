import re
import shlex
import subprocess
import threading
from typing import Optional

import fabric


def parse_job_id(output: str) -> int:
    match = re.search(r"job (?P<id>[0-9]+)", output)
    if match is None:
        raise ValueError(f"Unable to parse job-id from:\n{output}")
    return int(match.group("id"))


class Client:
    _connection: Optional[fabric.Connection] = None

    def __init__(
        self, hostname: Optional[str] = None, username: Optional[str] = None
    ) -> None:
        self._hostname = hostname
        self._username = username
        self._mutex = threading.Lock()
        self._connection = None
        if hostname is not None:
            self._connection = fabric.Connection(host=hostname, user=username)

    def close(self):
        with self._mutex:
            if self._connection is not None:
                self._connection.close()  # type: ignore

    def launch(self, command) -> int:
        output = self._submit_command(command)
        return parse_job_id(output)

    def _run_command(self, command: str) -> str:
        if self._connection is None:
            return subprocess.check_output(shlex.split(command), text=True)
        else:
            with self._mutex:
                return self._connection.run(command, hide="both").stdout

    def _submit_command(self, command: str) -> str:
        return self._run_command(f"sbatch {command}")

    def __repr__(self):
        if self._connection is not None:
            return f'Client(hostname="{self._hostname}", user="{self._username}")'
        else:
            return "Client()"
