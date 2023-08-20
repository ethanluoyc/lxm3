import re
import shlex
import subprocess
import threading
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import fabric


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


_KEYS = (
    "qname,hostname,group,owner,project,"
    "department,jobname,jobnumber,taskid,"
    "account,priority,qsub_time,start_time,"
    "end_time,granted_pe,slots,failed,exit_status,"
    "ru_wallclock,ru_utime,ru_stime,ru_maxrss,ru_ixrss,ru_ismrss,ru_idrss,ru_isrss,ru_minflt,"
    "ru_majflt,ru_nswap,ru_inblock,ru_oublock,ru_msgsnd,ru_msgrcv,ru_nsignals,ru_nvcsw,ru_nivcsw,"
    "cpu,mem,io,iow,maxvmem,arid,ar_sub_time,category"
).split(",")


def parse_accounting(data: str) -> List[Dict[str, str]]:
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
    _connection: Optional[fabric.Connection]

    def __init__(
        self, hostname: Optional[str] = None, username: Optional[str] = None
    ) -> None:
        self._hostname = hostname
        self._username = username
        self._connection = None
        self._mutex = threading.Lock()
        if hostname is not None:
            self._connection = fabric.Connection(host=hostname, user=username)
        self._qacct_conn = None

    def close(self):
        with self._mutex:
            if self._connection is not None:
                self._connection.close()
            if self._qacct_conn is not None:
                self._qacct_conn.close()

    def _run_command(self, command: str) -> str:
        if self._connection is None:
            return subprocess.check_output(shlex.split(command), text=True)
        else:
            with self._mutex:
                result = self._connection.run(command, hide="both")
                return result.stdout

    def launch(self, command):
        output = self._run_command(f"qsub {command}")
        match = parse_job_id(output)
        return match

    def qstat(self):
        stats = parse_qstat(self._run_command("qstat -xml"))
        return stats

    def cancel(self, job_id: str) -> None:
        self._run_command(f"qdel {job_id}")

    def qacct(self, job_id: Optional[str] = None):
        if self._qacct_conn is None:
            if self._connection is None:
                raise NotImplementedError("qacct with local session not implemented")
            sftp = self._connection.sftp()
            default_accounting_path = self._run_command(
                "echo $SGE_ROOT/$SGE_CELL/common/accounting"
            ).strip()
            try:
                sftp.stat(default_accounting_path)
            except IOError:
                act_qmaster_address = self._run_command(
                    "cat $SGE_ROOT/$SGE_CELL/common/act_qmaster"
                ).strip()
                qacct_conn = fabric.Connection(
                    act_qmaster_address, user=self._username, gateway=self._connection
                )
                self._qacct_conn = qacct_conn
            else:
                self._qacct_conn = self._connection

        with self._mutex:
            if not job_id:
                job_id = ""
            command = f"qacct -j {job_id} -u {self._username}"
            output = self._qacct_conn.run(command, hide="both").stdout
            return parse_accounting(output)

    def shell(self):
        if self._connection is None:
            raise NotImplementedError("shell with local session not implemented")
        self._connection.shell()

    def __repr__(self):
        if self._connection is not None:
            return f'Client(hostname="{self._hostname}", user="{self._username}")'
        else:
            return "Client()"
