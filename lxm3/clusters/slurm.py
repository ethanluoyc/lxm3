import logging
import re
import shlex
import subprocess
import threading
from typing import Optional

import paramiko

logging.getLogger("paramiko.transport").setLevel(logging.WARNING)


def parse_job_id(output: str) -> int:
    match = re.search(r"job (?P<id>[0-9]+)", output)
    if match is None:
        raise ValueError(f"Unable to parse job-id from:\n{output}")
    return int(match.group("id"))


class Client:
    def __init__(
        self, hostname: Optional[str] = None, username: Optional[str] = None
    ) -> None:
        self._hostname = hostname
        self._username = username
        self._ssh = None
        if hostname is not None:
            self._connect(hostname, username)
        self._qacct_ssh = None

    def _connect(self, hostname: str, username: Optional[str] = None):
        self._ssh = paramiko.SSHClient()
        self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
        self._ssh.load_system_host_keys()
        self._lock = threading.Lock()
        self._ssh.connect(hostname=hostname, username=username)

    def close(self):
        if self._ssh is not None:
            with self._lock:
                self._ssh.close()  # type: ignore
                if self._qacct_ssh is not None and self._qacct_ssh is not self._ssh:
                    self._qacct_ssh.close()

    def launch(self, command) -> int:
        output = self._submit_command(command)
        return parse_job_id(output)

    def _run_command(self, command: str) -> str:
        if self._ssh is None:
            return subprocess.check_output(shlex.split(command), text=True)
        else:
            with self._lock:
                _, stdout_, stderr_ = self._ssh.exec_command(command)  # type: ignore
                retcode = stdout_.channel.recv_exit_status()
                stderr = stderr_.read().decode()
                stdout = stdout_.read().decode()
                if retcode != 0:
                    raise RuntimeError(
                        f"Failed to run command: {command}\n"
                        f"stdout:{stdout}\n"
                        f"stderr:{stderr}"
                    )
                return stdout

    def _submit_command(self, command: str) -> str:
        return self._run_command(f"sbatch {command}")

    def __repr__(self):
        if self._ssh is not None:
            return f'Client(hostname="{self._hostname}", user="{self._username}")'
        else:
            return "Client()"
