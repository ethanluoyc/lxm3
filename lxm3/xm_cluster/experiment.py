import asyncio
import contextlib
import functools
import subprocess
import threading
import time
from concurrent import futures
from typing import Any, Awaitable, Callable, List, Mapping, Optional

import vcsinfo
from absl import logging

from lxm3._vendor.xmanager import xm
from lxm3._vendor.xmanager.xm import async_packager
from lxm3._vendor.xmanager.xm import core
from lxm3._vendor.xmanager.xm import id_predictor
from lxm3._vendor.xmanager.xm import job_blocks
from lxm3._vendor.xmanager.xm import pattern_matching as pm
from lxm3.xm_cluster import config as config_lib
from lxm3.xm_cluster import executors
from lxm3.xm_cluster import metadata
from lxm3.xm_cluster import packaging
from lxm3.xm_cluster.console import console
from lxm3.xm_cluster.execution import gridengine as gridengine_execution
from lxm3.xm_cluster.execution import local as local_execution
from lxm3.xm_cluster.execution import slurm as slurm_execution


def _gridengine_job_predicate(job: xm.Job):
    return isinstance(job.executor, executors.GridEngine)


def _slurm_job_predicate(job: xm.Job):
    return isinstance(job.executor, executors.Slurm)


def _local_job_predicate(job: xm.Job):
    return isinstance(job.executor, executors.Local)


class _LaunchResult:
    def __init__(self, local_handles, non_local_handles):
        self.local_handles = local_handles
        self.non_local_handles = non_local_handles


async def _launch(jobs: List[xm.Job]):
    experiment: ClusterExperiment = core._current_experiment.get()  # type: ignore
    gridengine_jobs = list(filter(_gridengine_job_predicate, jobs))
    slurm_jobs = list(filter(_slurm_job_predicate, jobs))
    local_jobs = list(filter(_local_job_predicate, jobs))

    if not (
        len(gridengine_jobs) == len(jobs)
        or len(local_jobs) == len(jobs)
        or len(slurm_jobs) == len(jobs)
    ):
        raise ValueError(
            "ClusterExperiment only supports running only GridEngine XOR "
            "Local XOR Slurm executors at the same time."
        )

    local_handles = []
    if local_jobs:
        local_handles.extend(
            await local_execution.launch(experiment._config, local_jobs)  # type: ignore
        )

    non_local_handles = []
    if gridengine_jobs:
        non_local_handles.extend(
            await gridengine_execution.launch(experiment._config, gridengine_jobs)  # type: ignore
        )

    if slurm_jobs:
        non_local_handles.extend(
            await slurm_execution.launch(experiment._config, slurm_jobs)  # type: ignore
        )

    return _LaunchResult(local_handles, non_local_handles)


class ClusterWorkUnit(xm.WorkUnit):
    """A mock version of WorkUnit with abstract methods implemented."""

    experiment: "ClusterExperiment"

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

        async with self._work_unit_id_predictor.submit_id(self._work_unit_id):  # type: ignore
            await self._submit_job_for_execution(job_group, args)

        # This is used by batched experiment to wait for all jobs to be launched.
        # before initiating a batch context
        self._launch_event.set()

    async def _submit_job_for_execution(self, job_group: xm.JobGroup, args):
        jobs: List[xm.Job] = list(job_group.jobs.values())  # type: ignore
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

    @property
    def context(self) -> metadata.ClusterMetadataContext:
        return metadata.ClusterMetadataContext()


class ClusterExperiment(xm.Experiment):
    """A mock version of Experiment with abstract methods implemented."""

    _async_packager = async_packager.AsyncPackager(packaging.package)

    def __init__(
        self,
        experiment_title: str,
        config: config_lib.Config,
        vcs: Optional[vcsinfo.VCS] = None,
    ) -> None:
        super().__init__()
        self.launched_jobs = []
        self.launched_jobs_args = []
        self._work_units = []
        self._experiment_id = int(time.time() * 10**3)
        self._in_batch_lock = threading.Lock()
        self._in_batch = False
        self.delayed_jobs = []
        self._experiment_title = experiment_title
        self._config = config
        self._vcs = vcs

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
                "Launching batch experiment from async context is not yet supported."
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
                console.print(
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

    @property
    def context(self) -> metadata.ClusterMetadataContext:
        return metadata.ClusterMetadataContext()


@functools.lru_cache()
def _load_vcsinfo() -> Optional[vcsinfo.VCS]:
    vcs = None

    try:
        vcs_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
        vcs = vcsinfo.detect_vcs(vcs_root)
    except subprocess.SubprocessError:
        logging.debug("Failed to detect VCS info")

    return vcs


def create_experiment(
    experiment_title: str, config: Optional[config_lib.Config] = None
) -> ClusterExperiment:
    """Create a LXM3 experiment backed by the xm_cluster backend.
    Args:
        experiment_title: Title of the experiment.
        config: Optional config object to use. If set, override
            the configuration loaded from the config file.
    """
    config = config or config_lib.default()
    vcs = _load_vcsinfo()

    if not config.project() and vcs is not None:
        config.set_project(vcs.name)

    return ClusterExperiment(experiment_title, config=config)
