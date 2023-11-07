# LXM3: XManager launch backend for HPC clusters
[![PyPI version](https://badge.fury.io/py/lxm3.svg)](https://badge.fury.io/py/lxm3)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lxm3)
![Read the Docs](https://img.shields.io/readthedocs/lxm3)
[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm-project.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

lxm3 provides an implementation for DeepMind's [XManager](https://github.com/deepmind/xmanager/tree/main) launch API that aims to provide a similar experience for running experiments on traditional HPC.

Currently, lxm3 provides a local execution backend and support for the [SGE](https://en.wikipedia.org/wiki/Oracle_Grid_Engine) and [Slurm](https://slurm.schedmd.com/) schedulers.

__NOTE__: lxm3 is still in early development. The API is not stable and may change in the future. We periodically update tag versions which are considered in good shape.
If you use lxm3 in your project, please pin to a specific commit/tag to avoid
breaking changes.

## Installation
For running on a cluster, you should install Singularity and rsync before using lxm3.
It may be possible to run on the cluster without Singularity, but that path
was not tested thoroughly.

You can install lxm3 from PyPI by running.
```bash
pip install lxm3
```
You can also install from GitHub for the latest features.
```bash
# Consider pinning to a specific commit/tag.
pip install git+https://github.com/ethanluoyc/lxm3
```

## Documentation
At a high level you can launch experiment by creating a launch script
called `launcher.py` that looks like:

```python
with xm_cluster.create_experiment(experiment_title="hello world") as experiment:
    # Launch on a Slurm cluster
    executor = xm_cluster.Slurm()
    # or, if you want to use SGE:
    # executor = xm_cluster.GridEngine()
    # or, if you want to run locally:
    # executor = xm_cluster.Local()

    spec = xm_cluster.PythonPackage(
       path=".",
       entrypoint=xm_cluster.ModuleName("my_package.main"),
    )

    # package your code
    [executable] = experiment.package(
        [xm.Packageable(spec, executor_spec=executor.Spec())]
    )

    # add jobs to your experiment
    experiment.add(
        xm.Job(executable=executable, executor=executor)
    )
```
and launch the experimet from the command line with
```python
lxm3 launch launcher.py
```

Many things happen under the hood. Since lxm3 implements the XManager
API, you should get familiar with the concepts in the
[XManager](https://github.com/deepmind/xmanager). Once you are
familiar with the concepts, checkout the [examples/](examples/)
directory for a quick start guide.


## Components
lxm3 provides the following executable specification and executors.

### Executable specifications
| Name      | Description |
| ----------- | ----------- |
| `lxm3.xm_cluster.PythonPackage`      | A python application packageable with pip |
| `lxm3.xm_cluster.UniversalPackage`      | A universal package |
| `lxm3.xm_cluster.SingularityContainer` | An executable running with Singularity |

### Executors
| Name      | Description |
| ----------- | ----------- |
| `lxm3.xm_cluster.Local`     | Runs a executable locally, mainly used for testing |
| `lxm3.xm_cluster.GridEngine`     | Runs a executable on SGE cluster |
| `lxm3.xm_cluster.Slurm`     | Runs a executable on Slurm cluster |

### Jobs
* Currently, only `xm.Job` and `xm.JobGenerator` that generates `xm.Job` are supported.
* We support HPC array jobs via `xm_cluster.ArrayJob`. See below.

## Implementation Details
### __Managing Dependencies with Containers__
lxm3 uses of Singularity containers for running jobs on HPCs.

lxm3 aims at providing a easy workflow for launching jobs on traditional HPC clusters,
which deviates from typical workflows for launching experiments on Cloud platforms.

lxm3 is designed for working with containerized applications using [Singularity](https://docs.sylabs.io/guides/3.5/user-guide/introduction.html) as the runtime.
Singularity is a popular choice for HPC clusters because it allows users to run containers without requiring root privileges, and is supported by many HPC clusters worldwide.

There are many benefits to using containers for running jobs on HPCs compared to traditional isolation via `venv` or `conda`.

1. `venv` and `conda` are laid out as a directory of files on the environment. For many
HPCs, normally these will be installed on a networked filesystem such as NFS. Operations
on these virtual environments are slow and inefficient. For example, on our cluster, removing
a `conda` environment with many dependencies can take an hour when these environments are
on NFS. There are usually quota put in places not only for the file sizes but also the number of files.
For ML projects that uses depends on many (large) packages such as TensorFlow, PyTorch, it is very easy
to hit the quota limit. Singularity containers are a single file. This is both easy
for deployment and also avoids the file number quota.
2. Containers provide consistent environment for running jobs on different clusters as well as making it easy to use system dependencies not installed on HPC's host environment.

### __Automated Deployment__.
HPC deployments normally use a filesystem that are detached from the filesystems of the user's workstation.
Many tutorials for running jobs on HPCs request the users to either clone their repository on the login node or ask the users manually copy files to the cluster. Doing this repeatedly is tedious. lxm3 automates the deployments from your workstation to the HPC cluster so that you can do most of your work locally without having
directly login into the cluster.

Unlike Docker or other OCI images that are composed of multiple layers,
the Singularity Image Format (SIF) used by Singularity is a single file that contains the entire filesystem of the container. While this is convenient as deployment to a remote cluster can be performed with a single `scp/rsync` command. The lack of layer caching/sharing makes repeated deployments slow and inefficient.
For this reason, unlike typical cloud deployments where the application and dependencies are packaged into a single image, lxm3 uses a two-stage packaging process to separate the application and dependencies.
This allows applications with heavy dependencies to be packaged once and reused across multiple experiments by reusing the same singularity container.

For Python applications, we rely on the user to first build a runtime image for all of the dependencies and use [standard Python pacakging](https://packaging.python.org/en/latest/tutorials/packaging-projects/) tools to create a distribution that is deployed separately to the cluster.
Concretely, the user is expected to create a simple `pyproject.toml` which describes how to create a distribution for their applications.
This is convenient, as lxm3 does not have to invent a custom packaging format for Python applications. For example, a simple `pyproject.toml` that uses `hatchling` as the build backend looks like:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "py_package"
authors = [{ name = "Joe" }]
version = "0.1.0"
```
lxm3 uses `pip install --no-deps` to create a zip archive that contains all of your application code.
In addition to the packaging, lxm3 also allows you to carry extra files for your deployment via
`xm_cluster.Fileset`. This is for example useful if you want to deploy configuration files.

The zip archive is automatically extracted into a temporary directory on the cluster and executed from there.
Using a zip archive again minimizes the number of files that are deployed to the cluster so that you are
less likely to hit a file number limit.

If you are using a different language or you cannot package your python application easily
with standard Python packaging tools, you can use `xm_cluster.UniversalPackage` to package your application.

### __Easy Hyperparameter Sweeping__.
For many scientific research projects, it's common to run the same experiment with different hyperparameters. lxm3 automatically generates jobs scripts that can be submitted to the cluster's scheduler for running multiple experiments with different hyperparameters passed as differnt command line arguments or environment variables.

For large parameter sweep, launching many separate jobs at once can
overwhelm the scheduler. For this reason, HPC schedulers encourage the
use of job arrays to submit sweeps. lxm3 provide a `ArrayJob`
xm.JobConfig that allows you to submit multiple jobs with the same
executable and job requirements but different hyperparameters as a
single job array.

For example:
```python
from lxm3 import xm
from lxm3 import xm_cluster
with xm_cluster.create_experiment() as experiment:
    executable = ...
    executor = ...
    parameters = [{"seed": seed} for seed in range(5)]
    experiment.add(
        xm_cluster.ArrayJob(executable=executable, executor=executor, args=parameters)
    )
```
This will be translated as passing `--seed {0..4}` to your executable. We
also support customzing environment variables, which is convenient for example if you
use [Weights and Biases](https://wandb.ai/site) where you can configure run names and groups
from environment variables (TODO(yl): migrate examples for configuring wandb).

There is a lot of flexibility on how to create the `args` for each job.
For example, you can use `itertools.product` to create a cartesian product of all the hyperparameters.
You can create arbitrary sweeps in pure python without resorting to a DSL.

Under the hood, lxm3 automatically generates job scripts that map from the array job index
to command line arguments and environment variables.

### __Separate experiment launching from your application__.
Similar to the design of XManager, lxm3 separates the launching of experiments from your application
so that you are not bound to a specific experiment framework.
You can develop your project with your favorite framework without having your application
code be aware of the existence of lxm3. In fact, we recommend that you install lxm3
as a development dependency that are not bundled with your dependencies used at runtime.
You can also install lxm3 globally or in its own virtual environment via `pex` or `pipx`.

### Notes for existing Xmanager users

1. We vendored a copy of xmanager core API (v0.4.0) into lxm3 with
light modification to support Python 3.9. This also allows us to just use the launch API
without `xm_local`'s dependencies. Thus, you should import the core API as
`from lxm3 import xm` instead of `from xmanager import xm`. Our executables specs
are defined in `xm_cluster` instead `xm` as we do not support the executable specs
from the core API.
