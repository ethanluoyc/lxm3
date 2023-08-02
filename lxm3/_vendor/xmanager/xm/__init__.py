# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""XManager client API.

Provides XManager public API for configuring and launching experiments.
"""

from lxm3._vendor.xmanager.xm import job_operators
from lxm3._vendor.xmanager.xm.compute_units import *
from lxm3._vendor.xmanager.xm.core import AuxiliaryUnitJob
from lxm3._vendor.xmanager.xm.core import AuxiliaryUnitRole
from lxm3._vendor.xmanager.xm.core import Experiment
from lxm3._vendor.xmanager.xm.core import ExperimentUnit
from lxm3._vendor.xmanager.xm.core import ExperimentUnitError
from lxm3._vendor.xmanager.xm.core import ExperimentUnitFailedError
from lxm3._vendor.xmanager.xm.core import ExperimentUnitNotCompletedError
from lxm3._vendor.xmanager.xm.core import ExperimentUnitRole
from lxm3._vendor.xmanager.xm.core import ExperimentUnitStatus
from lxm3._vendor.xmanager.xm.core import Importance
from lxm3._vendor.xmanager.xm.core import LaunchedJob
from lxm3._vendor.xmanager.xm.core import NotFoundError
from lxm3._vendor.xmanager.xm.core import WorkUnit
from lxm3._vendor.xmanager.xm.core import WorkUnitCompletedAwaitable
from lxm3._vendor.xmanager.xm.core import WorkUnitRole
from lxm3._vendor.xmanager.xm.executables import BazelBinary
from lxm3._vendor.xmanager.xm.executables import BazelContainer
from lxm3._vendor.xmanager.xm.executables import Binary
from lxm3._vendor.xmanager.xm.executables import BinaryDependency
from lxm3._vendor.xmanager.xm.executables import CommandList
from lxm3._vendor.xmanager.xm.executables import Container
from lxm3._vendor.xmanager.xm.executables import Dockerfile
from lxm3._vendor.xmanager.xm.executables import ModuleName
from lxm3._vendor.xmanager.xm.executables import PythonContainer
from lxm3._vendor.xmanager.xm.job_blocks import Constraint
from lxm3._vendor.xmanager.xm.job_blocks import Executable
from lxm3._vendor.xmanager.xm.job_blocks import ExecutableSpec
from lxm3._vendor.xmanager.xm.job_blocks import Executor
from lxm3._vendor.xmanager.xm.job_blocks import ExecutorSpec
from lxm3._vendor.xmanager.xm.job_blocks import get_args_for_all_jobs
from lxm3._vendor.xmanager.xm.job_blocks import Job
from lxm3._vendor.xmanager.xm.job_blocks import JobConfig
from lxm3._vendor.xmanager.xm.job_blocks import JobGeneratorType
from lxm3._vendor.xmanager.xm.job_blocks import JobGroup
from lxm3._vendor.xmanager.xm.job_blocks import JobType
from lxm3._vendor.xmanager.xm.job_blocks import merge_args
from lxm3._vendor.xmanager.xm.job_blocks import Packageable
from lxm3._vendor.xmanager.xm.job_blocks import SequentialArgs
from lxm3._vendor.xmanager.xm.job_blocks import UserArgs
from lxm3._vendor.xmanager.xm.metadata_context import ContextAnnotations
from lxm3._vendor.xmanager.xm.metadata_context import MetadataContext
from lxm3._vendor.xmanager.xm.packagables import bazel_binary
from lxm3._vendor.xmanager.xm.packagables import bazel_container
from lxm3._vendor.xmanager.xm.packagables import binary
from lxm3._vendor.xmanager.xm.packagables import container
from lxm3._vendor.xmanager.xm.packagables import dockerfile_container
from lxm3._vendor.xmanager.xm.packagables import python_container
from lxm3._vendor.xmanager.xm.resources import GpuType
from lxm3._vendor.xmanager.xm.resources import InvalidTpuTopologyError
from lxm3._vendor.xmanager.xm.resources import JobRequirements
from lxm3._vendor.xmanager.xm.resources import ResourceDict
from lxm3._vendor.xmanager.xm.resources import ResourceQuantity
from lxm3._vendor.xmanager.xm.resources import ResourceType
from lxm3._vendor.xmanager.xm.resources import ServiceTier
from lxm3._vendor.xmanager.xm.resources import Topology
from lxm3._vendor.xmanager.xm.resources import TpuType
from lxm3._vendor.xmanager.xm.utils import run_in_asyncio_loop
from lxm3._vendor.xmanager.xm.utils import ShellSafeArg
