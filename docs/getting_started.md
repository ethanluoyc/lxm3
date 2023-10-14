# Getting Started

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

## Understanding the XManager API
Since lxm3 implements the XManager API, you should get familiar with
the concepts in the [XManager](https://github.com/deepmind/xmanager).

## Examples
Once you are familiar with the concepts, checkout the
[examples/](https://github.com/ethanluoyc/lxm3/tree/main/examples)
directory for a quick start guide.
