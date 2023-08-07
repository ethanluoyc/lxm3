import asyncio
import contextlib
import os
import threading
import time
from concurrent import futures
from typing import Any, Awaitable, Callable, List, Mapping, Optional

import fsspec

from lxm3._vendor.xmanager import xm
from lxm3._vendor.xmanager.xm import async_packager
from lxm3._vendor.xmanager.xm import core
from lxm3._vendor.xmanager.xm import id_predictor
from lxm3._vendor.xmanager.xm import job_blocks
from lxm3._vendor.xmanager.xm import pattern_matching as pm
from lxm3.xm_cluster import config as config_lib
from lxm3.xm_cluster import executors
from lxm3.xm_cluster import packaging
from lxm3.xm_cluster.execution import gridengine as gridengine_execution
from lxm3.xm_cluster.execution import local as local_execution


def _gridengine_job_predicate(job: xm.Job):
    return isinstance(job.executor, executors.GridEngine)


def _local_job_predicate(job: xm.Job):
    return isinstance(job.executor, executors.Local)


class _LaunchResult:
    def __init__(self, local_handles, non_local_handles):
        self.local_handles = local_handles
        self.non_local_handles = non_local_handles


async def _launch(jobs):
    experiment: ClusterExperiment = core._current_experiment.get()
    gridengine_jobs = list(filter(_gridengine_job_predicate, jobs))
    local_jobs = list(filter(_local_job_predicate, jobs))
    if not (len(gridengine_jobs) == len(jobs) or len(local_jobs) == len(jobs)):
        raise ValueError(
            "ClusterExperiment only supports running only GridEngine XOR "
            "Local executors at the same time."
        )

    local_handles = []
    if local_jobs:
        storage_root = os.path.abspath(experiment._local_staging_directory)
        fs = fsspec.filesystem("file")
        local_handles.extend(await local_execution.launch(fs, storage_root, local_jobs))

    non_local_handles = []
    if gridengine_jobs:
        storage_root = experiment._cluster_staging_directory
        fs = fsspec.filesystem(
            "sftp", host=experiment._cluster_hostname, username=experiment._cluster_user
        )

        # Normalize the storage root to an absolute path.
        if not os.path.isabs(storage_root):
            storage_root = fs.ftp.normalize(storage_root)

        non_local_handles.extend(
            await gridengine_execution.launch(
                experiment._cluster_hostname,
                experiment._cluster_user,
                fs,
                storage_root,
                jobs,
            )
        )
    return _LaunchResult(local_handles, non_local_handles)


class ClusterWorkUnit(xm.WorkUnit):
    """A mock version of WorkUnit with abstract methods implemented."""

    def __init__(
        self,
        experiment: "ClusterExperiment",
        work_unit_id_predictor: id_predictor.Predictor,
        create_task: Callable[[Awaitable[Any]], futures.Future[Any]],
        launched_jobs: List[job_blocks.JobType],
        launched_jobs_args: List[Optional[Mapping[str, Any]]],
        args: Optional[Mapping[str, Any]],
        role: xm.ExperimentUnitRole,
    ) -> None:
        super().__init__(experiment, create_task, args, role)
        self._launched_jobs = launched_jobs
        self._launched_jobs_args = launched_jobs_args
        self._work_unit_id = work_unit_id_predictor.reserve_id()
        self._work_unit_id_predictor = work_unit_id_predictor
        self._launch_event = threading.Event()
        self._local_handles = []
        self._non_local_handles = []

    async def _launch_job_group(
        self,
        job_group: xm.JobGroup,
        args: Optional[Mapping[str, Any]],
        identity: str,
    ) -> None:
        """Appends the job group to the launched_jobs list."""
        del identity

        async with self._work_unit_id_predictor.submit_id(self._work_unit_id):
            await self._submit_job_for_execution(job_group.jobs, args)

        # This is used by batched experiment to wait for all jobs to be launched.
        # before initiating a batch context
        self._launch_event.set()

    async def _submit_job_for_execution(self, jobs: xm.JobGroup, args):
        jobs = list(jobs.values())
        assert len(jobs) == 1
        if self.experiment.is_in_batch():

            def callback(result):
                self._ingest_handles(result)

            assert len(jobs) == 1
            self.experiment._register_delayed_job((jobs[0], args), callback)
        else:
            launch_result = await _launch(jobs)
            self._ingest_handles(launch_result)

    def _ingest_handles(self, launch_result):
        """"""
        self._local_handles.extend(launch_result.local_handles)
        self._non_local_handles.extend(launch_result.non_local_handles)

    async def wait_for_local_jobs(self, is_exit_abrupt: bool):
        if not is_exit_abrupt:
            await asyncio.gather(*[handle.wait() for handle in self._local_handles])

    @property
    def work_unit_id(self) -> int:
        return self._work_unit_id

    @property
    def experiment_unit_name(self) -> str:
        return f"{self.experiment_id}_{self._work_unit_id}"


class ClusterExperiment(xm.Experiment):
    """A mock version of Experiment with abstract methods implemented."""

    _async_packager = async_packager.AsyncPackager(packaging.package)

    def __init__(
        self,
        experiment_title: str,
        local_staging_directory,
        cluster_hostname: str,
        cluster_user: str,
        cluster_staging_directory: str,
    ) -> None:
        super().__init__()
        self.launched_jobs = []
        self.launched_jobs_args = []
        self.delayed_jobs = []
        self._work_units = []
        self._experiment_id = int(time.time() * 10**3)
        self._in_batch_lock = threading.Lock()
        self._in_batch = False
        self._experiment_title = experiment_title
        self._local_staging_directory = local_staging_directory
        self._cluster_hostname = cluster_hostname
        self._cluster_user = cluster_user
        self._cluster_staging_directory = cluster_staging_directory

    def is_in_batch(self):
        with self._in_batch_lock:
            return self._in_batch

    @contextlib.contextmanager
    def batch(self):
        is_coro_context = False
        try:
            asyncio.get_running_loop()
            is_coro_context = True
        except RuntimeError:
            pass
        if is_coro_context:
            raise RuntimeError(
                "When using Experiment.batch from a coroutine please use "
                "`async with experiment.async_batch` syntax"
            )
        try:
            assert not self._in_batch
            if len(self._work_units) > 0:
                self._work_units[-1]._launch_event.wait()
            self._in_batch = True
            yield
        finally:
            if len(self._work_units) > 0:
                self._work_units[-1]._launch_event.wait()
            delayed_jobs = [j[0] for j in self.delayed_jobs]
            delayed_cb = [j[2] for j in self.delayed_jobs]

            async def launch_array_jobs():
                array_launch_result = await _launch(delayed_jobs)
                results = []
                if array_launch_result.local_handles:
                    results = [_LaunchResult(array_launch_result.local_handles, [])]
                else:
                    results = [_LaunchResult([], array_launch_result.non_local_handles)]

                for callback, results in zip(delayed_cb, results):
                    callback(results)

            self._create_task(launch_array_jobs())
            self.delayed_jobs = []
            self._in_batch = False

    @contextlib.asynccontextmanager
    async def async_batch(self):
        try:
            assert not self._in_batch
            if len(self._work_units) > 0:
                while not self._work_units[-1]._launch_event.is_set():
                    await asyncio.sleep(0.1)
            self._in_batch = True
            yield
        finally:
            if len(self._work_units) > 0:
                while not self._work_units[-1]._launch_event.is_set():
                    await asyncio.sleep(0.1)
            delayed_jobs = [j[0] for j in self.delayed_jobs]
            delayed_cb = [j[2] for j in self.delayed_jobs]
            handles = _launch(delayed_jobs)
            for callback, handle in zip(delayed_cb, handles):
                callback(handle)
            self.delayed_jobs = []
            self._in_batch = False

    def _register_delayed_job(self, job_and_args, callback):
        job, args = job_and_args
        self.delayed_jobs.append((job, args, callback))

    def _create_experiment_unit(
        self,
        args: Optional[Mapping[str, Any]],
        role: xm.ExperimentUnitRole = xm.WorkUnitRole(),
        identity: str = "",
    ) -> Awaitable[ClusterWorkUnit]:
        """Creates a new WorkUnit instance for the experiment."""
        del identity  # Unused.
        future = asyncio.Future(loop=self._event_loop)
        experiment_unit = ClusterWorkUnit(
            self,
            self._work_unit_id_predictor,
            self._create_task,
            self.launched_jobs,
            self.launched_jobs_args,
            args,
            role,
        )

        def _unsupported_aux_units(_):
            raise NotImplementedError("Auxiliary units are not supported")

        pm.match(
            pm.Case(
                [xm.WorkUnitRole],
                lambda _: self._work_units.append(experiment_unit),
            ),
            pm.Case(
                [xm.AuxiliaryUnitRole],
                _unsupported_aux_units,
            ),
        )(role)

        future.set_result(experiment_unit)
        return future

    def _wait_for_local_jobs(self, is_exit_abrupt: bool):
        if self._work_units:
            if any([wu._local_handles for wu in self._work_units]):
                print(
                    "Waiting for local jobs to complete. "
                    "Press Ctrl+C to terminate them and exit"
                )
        for unit in self._work_units:
            self._create_task(unit.wait_for_local_jobs(is_exit_abrupt))

    def __exit__(self, exc_type, exc_value, traceback):
        # Flush `.add` calls.
        self._wait_for_tasks()
        self._wait_for_local_jobs(exc_value is not None)
        return super().__exit__(exc_type, exc_value, traceback)

    async def __aexit__(self, exc_type, exc_value, traceback):
        # Flush `.add` calls.
        await self._await_for_tasks()
        self._wait_for_local_jobs(exc_value is not None)
        return await super().__aexit__(exc_type, exc_value, traceback)

    @property
    def work_unit_count(self) -> int:
        return len(self.work_units)

    @property
    def work_units(self):
        return self._work_units

    @property
    def experiment_id(self) -> int:
        return self._experiment_id


def create_experiment(
    experiment_title: str = "",
    *,
    project=None,
    cluster=None,
    config=None,
):
    config = config or config_lib.default()
    if cluster is not None:
        for cluster_config in config["clusters"]:
            if cluster_config["name"] == cluster:
                break
        raise ValueError(f"Cluster {cluster} not found in config")
    else:
        cluster_config = config["clusters"][0]

    if project is None:
        project = config["project"]

    return ClusterExperiment(
        experiment_title,
        local_staging_directory=os.path.join(
            config["local"]["storage"]["staging"], project
        ),
        cluster_hostname=cluster_config["server"],
        cluster_user=cluster_config["user"],
        cluster_staging_directory=os.path.join(
            cluster_config["storage"]["staging"], project
        ),
    )
