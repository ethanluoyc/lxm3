# Basic example
This example shows how to launch python programs in JAX (with GPU acceleration) using
lxm3.

## 1. Prerequisites
This section walks through the steps for setting up Singularity, which is the preferred
way of running jobs created by lxm3.

This section is not specific to lxm3: you can use the same steps to set up Singularity
for other projects.

### 1.1 Install Singularity
The example uses a Singularity container for installing all dependencies. You can
install SingularityCE by following the [official
documentaion](https://docs.sylabs.io/guides/latest/user-guide/). There is a simple
installation script for Ubuntu in the [docs](../../docs/install-singularity.sh) folder.

An example installation session looks like:
```bash
ARCH='amd64'
VERSION='3.11.4'
CODENAME=$(lsb_release -s -c)
SINGULARITY_URL="https://github.com/sylabs/singularity/releases/download/v${VERSION}/singularity-ce_${VERSION}-${CODENAME}_${ARCH}.deb"

curl -O -L $SINGULARITY_URL
curl -O -L "https://github.com/sylabs/singularity/releases/download/v${VERSION}/sha256sums"

# Install (requires sudo)
sudo apt-get install $PWD/$(basename $SINGULARITY_URL)

```

In this example, we will use a pre-built Docker container. Singularity will convert
the Docker container to a Singularity's SIF format automatically. This assumes that
docker is installed on your machine. It's quite useful to have docker installed
even if you don't use it directly as there it enables you to use pre-built many containers which are available on DockerHub.

Let's check that singularity is installed correctly by running a simple container.
```bash
# Pulling from a Docker container
singularity pull docker://godlovedc/lolcow
singularity exec lolcow_latest.sif cowsay moo
# Should print:
#  _____
# < moo >
#  -----
#         \   ^__^
#          \  (oo)\_______
#             (__)\       )\/\
#                 ||----w |
#                 ||     ||
```

### 1.2 Build a singularity container

We will demonstrate how to build a singularity container from a Docker container in this
example. This is convenient if you already know how to write a Dockerfile.
Alternatively, you can also build from a Singularity definition file.

We have created a [Makefile](./Makefile) to automate this process. Have a look at it to
understand what's happening under the hood. For now, let's just run it.

```bash
make build-singularity
```
There should now be a new file called `jax-cuda.sif` in the current directory. You can
run it, test it out to get familiar with it. Note that if you want to run the container
with access to your host GPU, you need to add the `--nv` flag. For example, to open a
shell session with the container, you can run:
```bash
singularity shell --nv jax-cuda.sif
# You are now inside the shell in the container,
# try running the `nvidia-smi` command
Singularity> nvidia-smi
```

If you don't have docker installed, there is a Singularity definition file for this
example. You can build the container from the definition file by running:
```bash
sudo singularity build jax-cuda.sif Singularity
```

Generally, we recommend using Dockerfile (or podman) for building a container and
convert that into a Singularity image. docker is likely more widely used and that allows
others to reproduce your environment even if they have no experience with singularity.

Great! You have now completed all pre-requisites for working with Singularity images.
Let's move on to the next section.

## 2. Running the example with LXM3

### 2.1 Install lxm3
```bash
python3 -m venv .venv
source .venv/bin/activate
# Install lxm3
python3 -m pip install ../../
# Try out the lxm3 command
lxm3 --help
```

### 2.2 Set up a configuration file for lxm3
Put the following content in a file called `lxm.toml`:
```toml
project = "" # Optional project name
# Configuration for running in local mode.
[local]
[local.storage]
staging = ".cache/lxm"

# Configuration for running on clusters.
[[clusters]]
# Set a name for this cluster, e.g., "cs"
name = "<TODO>"
# Replace with the server you normally use for ssh into the cluster, e.g. "beaker.cs.ucl.ac.uk"
server = "<TODO>"
# Fill in the username you use for this cluster.
user = "<TODO>"

[clusters.storage]
# Replace with the path to a staging directory on the cluster. lxm3 uses this directory for storing all files required to run your job.
staging = "<absolute path to your home directory>/lxm3-staging"

```
### 2.3 Run the example
Before explaining the details, let's just run the example and see what happens.
#### Running locally
Let's first try running the example locally.
```bash
lxm3 launch launcher.py -- --lxm_config lxm.toml
```

#### Running on the cluster
_Note: this example was tested on the [HPC cluster](https://hpc.cs.ucl.ac.uk/) from the
CS department. Unfortunately, different SGE clusters have different ways of specifying
job requirements so you may have to modify the `JobRequirements` in `launcher.py` to
make it work on a different cluster._

Now let's try running the example on the cluster.
```bash
lxm3 launch launcher.py -- --lxm_config lxm.toml --launch_on_cluster
```

### 2.4 Run a batch job
```bash
lxm3 launch batch_launcher.py -- --lxm_config lxm.toml --launch_on_cluster
```
