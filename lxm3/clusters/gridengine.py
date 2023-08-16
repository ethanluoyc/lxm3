import logging
import re
import shlex
import subprocess
import threading
import xml.etree.ElementTree as ET
from typing import List, Optional

import paramiko

logging.getLogger("paramiko.transport").setLevel(logging.WARNING)


def parse_job_id(output: str) -> re.Match:
    match = re.search(
        r"(?P<job_id>\d+)(.?(?P<task_start>\d+)-(?P<task_end>\d+):(?P<task_step>\d+))?",
        output,
    )
    if match is None:
        raise ValueError(f"Unable to parse job-id from:\n{output}")
    return match


def split_job_ids(match: re.Match) -> List[str]:
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


def parse_qstat(qstat_xml_output: str):
    root = ET.fromstring(qstat_xml_output)
    infos = {}
    for joblist in root.iter("job_list"):
        job_number_elem = joblist.find("JB_job_number")
        if job_number_elem is None:
            raise ValueError()
        job_id = job_number_elem.text
        task_ids = joblist.find("tasks")
        if task_ids is not None:
            task_ids = task_ids.text
            if ":" in task_ids:  # type: ignore
                task_range, task_step = task_ids.split(":")  # type: ignore
                task_step = int(task_step)
                task_start, task_end = map(int, task_range.split("-"))
                for tid in range(task_start, task_end + 1, task_step):
                    infos[f"{job_id}.{tid}"] = {"state": joblist.attrib["state"]}
            elif "," in task_ids:  # type: ignore
                task_ids = map(int, task_ids.split(","))  # type: ignore
                for tid in task_ids:
                    infos[f"{job_id}.{tid}"] = {"state": joblist.attrib["state"]}
            else:
                tid = int(task_ids)  # type: ignore
                infos[f"{job_id}.{tid}"] = {"state": joblist.attrib["state"]}
        else:
            infos[job_id] = {"state": joblist.attrib["state"]}
    return infos


def parse_detailed_qstat(qstat_xml_output: str):
    root = ET.fromstring(qstat_xml_output)
    infos = {}
    for joblist in root.iter("job_list"):
        job_number_elem = joblist.find("JB_job_number")
        if job_number_elem is None:
            raise ValueError()
        job_id = job_number_elem.text
        task_ids = joblist.find("tasks")
        if task_ids is not None:
            task_ids = task_ids.text
            if ":" in task_ids:  # type: ignore
                task_range, task_step = task_ids.split(":")  # type: ignore
                task_step = int(task_step)
                task_start, task_end = map(int, task_range.split("-"))
                for tid in range(task_start, task_end + 1, task_step):
                    infos[f"{job_id}.{tid}"] = {"state": joblist.attrib["state"]}
            elif "," in task_ids:  # type: ignore
                task_ids = map(int, task_ids.split(","))  # type: ignore
                for tid in task_ids:
                    infos[f"{job_id}.{tid}"] = {"state": joblist.attrib["state"]}
            else:
                tid = int(task_ids)  # type: ignore
                infos[f"{job_id}.{tid}"] = {"state": joblist.attrib["state"]}
        else:
            infos[job_id] = {"state": joblist.attrib["state"]}
    return infos


_KEYS = (
    "qname,hostname,group,owner,project,"
    "department,jobname,jobnumber,taskid,"
    "account,priority,qsub_time,start_time,"
    "end_time,granted_pe,slots,failed,exit_status,"
    "ru_wallclock,ru_utime,ru_stime,ru_maxrss,ru_ixrss,ru_ismrss,ru_idrss,ru_isrss,ru_minflt,"
    "ru_majflt,ru_nswap,ru_inblock,ru_oublock,ru_msgsnd,ru_msgrcv,ru_nsignals,ru_nvcsw,ru_nivcsw,"
    "cpu,mem,io,iow,maxvmem,arid,ar_sub_time,category"
).split(",")


def parse_accounting(data):
    records = []
    record = {}
    for i, line in enumerate(data.split("\n")):
        if line.startswith("="):
            if record:
                records.append(record)
            record = {}
        else:
            tokens = line.strip().split(maxsplit=1)
            if len(tokens) == 2:
                key, value = tokens
                if key in _KEYS:
                    record[key] = value
    if record:
        records.append(record)
    return records


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
        if self._ssh is None:
            with self._lock:
                self._ssh.close()  # type: ignore
                if self._qacct_ssh is not None and self._qacct_ssh is not self._ssh:
                    self._qacct_ssh.close()

    def launch(self, command):
        output = self._submit_command(command)
        match = parse_job_id(output)
        return match

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
        return self._run_command(f"qsub {command}")

    def _stat_command(self) -> str:
        output = self._run_command("qstat -xml")
        return output

    def acct_command(self, job_id: Optional[str] = None):
        if not job_id:
            job_id = ""
        return f"qacct -j {job_id} -u {self._username}"

    def qstat(self):
        stats = parse_qstat(self._stat_command())
        return stats

    def qacct(self, job_id: Optional[str] = None):
        if self._qacct_ssh is None:
            if self._ssh is None:
                raise NotImplementedError("qacct with local session not implemented")
            sftp = self._ssh.open_sftp()
            default_accounting_path = self._run_command(
                "echo $SGE_ROOT/$SGE_CELL/common/accounting"
            ).strip()
            try:
                sftp.stat(default_accounting_path)
            except IOError:
                qacct_ssh_client = paramiko.SSHClient()
                qacct_ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                act_qmaster_address = self._run_command(
                    "cat $SGE_ROOT/$SGE_CELL/common/act_qmaster"
                ).strip()
                gateway = self._ssh
                sock = gateway.get_transport().open_channel(  # type: ignore
                    "direct-tcpip", (act_qmaster_address, 22), ("", 0)
                )
                qacct_ssh_client.connect(
                    act_qmaster_address, username=self._username, sock=sock
                )
                self._qacct_ssh = qacct_ssh_client
            else:
                self._qacct_ssh = self._ssh
            finally:
                sftp.close()

        with self._lock:
            _, stdout_, stderr_ = self._qacct_ssh.exec_command(self.acct_command(job_id))  # type: ignore
            retcode = stdout_.channel.recv_exit_status()
            stderr = stderr_.read().decode()
            stdout = stdout_.read().decode()
            if retcode != 0:
                if "not found" in stderr:
                    return []
                else:
                    raise RuntimeError(stderr)
            return parse_accounting(stdout)

    def _cancel_command(self, job_id: str) -> str:
        return self._run_command(f"qdel {job_id}")

    def cancel(self, job_id: str) -> None:
        self._cancel_command(job_id)

    def shell(self):
        import fabric

        conn = fabric.Connection(self._hostname, user=self._username)
        conn.shell()

    def __repr__(self):
        if self._ssh is not None:
            return f'Client(hostname="{self._hostname}", user="{self._username}")'
        else:
            return "Client()"
