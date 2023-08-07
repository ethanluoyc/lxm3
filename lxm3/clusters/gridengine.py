import re
import threading

import paramiko


def _extract_job_id(output):
    match = re.search(
        r"(?P<job_id>\d+)(.?(?P<task_start>\d+)-(?P<task_end>\d+):(?P<task_step>\d+))?",
        output,
    )
    if match is None:
        raise ValueError(f"Unable to parse job-id from:\n{output}")
    return match


def _split_job_ids(match):
    job_id = match.group("job_id")
    task_start = match.group("task_start")
    task_end = match.group("task_end")
    task_step = match.group("task_step")
    if not task_start:
        return [job_id]
    else:
        return [
            f"{job_id}.{tid}"
            for tid in range(int(task_start), int(task_end) + 1, int(task_step))
        ]


class Client:
    def __init__(self, hostname, username) -> None:
        self._hostname = hostname
        self._username = username
        self._ssh = None
        self._connect()

    def _connect(self):
        self._ssh = paramiko.SSHClient()
        self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
        self._ssh.load_system_host_keys()
        self._lock = threading.Lock()
        self._ssh.connect(hostname=self._hostname, username=self._username)

    def close(self):
        with self._lock:
            self._ssh.close()  # type: ignore

    def launch(self, command):
        output = self._submit_command(command)
        match = _extract_job_id(output)
        print(f"Successfully launched job {match.group(0)}")
        job_ids = _split_job_ids(match)
        return job_ids

    def _run_command(self, command):
        with self._lock:
            _, stdout, _ = self._ssh.exec_command(command)  # type: ignore
            stdout = stdout.read().decode()
            return stdout

    def _submit_command(self, command):
        return self._run_command(f"qsub {command}")

    def _cancel_command(self, job_id):
        return self._run_command(f"qdel {job_id}")

    def cancel(self, job_id: str):
        self._cancel_command(job_id)
