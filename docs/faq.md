# FAQs

__Running lxm3 launch raises `ValueError("Unable to load Config")`__

You should make sure that you have created lxm3 configuration file.

----

__Packaging Error when calling `experiment.package`__
LXM3 runs `pip install` to install your project (without dependnecies)
into a temporary directory and then packages the project into a zip file.
`Packaging Error` means that calling `pip install` on your project has failed.

The terminal output should contain the error message when calling `pip`
to package your project.

One way to debug this is to run `pip install` outside the launcher script
to see if your project can be installed properly with `pip`.

__lxm3 is unable to find my Python modules__

This normally means that your Python project is not packaged properly.

LXM3 relies on standard Python packaging tools to find your packages
and create a zip archive that will be extracted during runtime and added
the packages to your `PYTHONPATH`.

Therefore, it's important that you understand how your build system
works and how it finds your packages. If this is the first time you
are packaging your Python project, you may want to learn more about it
from [Python Packaging](https://packaging.python.org/).

One way to check that your Python packages are packaged properly is to
check the contents of wheel file created from your project. You can do
so by running
```bash
pip wheel --no-deps .
```
and then check the contents of the wheel file created for example by
running
```bash
unzip -t <path to your wheel>
```

You should be able to see your Python packages in the wheel file, in
installed layout. Also check if _data files (configuration files,
assets)_ are included in the wheel. If that is not the case, you
should consult your build system documentation for how to specify the
packages to be included in your project.

Here is a list of python build systems and their documentation how to
specify files to be packaged.

| Backend     |Documentation                                                                                                                                  |
|-------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| setuptools  | [https://setuptools.pypa.io/en/latest/userguide/package_discovery.html](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html) |
| hatchling   | [https://hatch.pypa.io/latest/build/](https://hatch.pypa.io/latest/build/)                                                                     |
| pdm-backend | [https://backend.pdm-project.org/build_config/](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html)                         |
| flit-core   | [https://flit.pypa.io/en/stable/pyproject_toml.html](https://flit.pypa.io/en/stable/pyproject_toml.html)                                       |

____

__How to check job logs when running on a HPC cluster__

When you launch on job on the cluster, the terminal should print out a
path where the job logs are stored. You can ssh into the cluster and
inspect that directory which would contain the job logs.

----

__How to rerun a failed job__

Currently, LXM3 does not have native support for rerunning failed
jobs. But since LXM3 generates a job script, you can rerun the job by
`qsub` the job script directly. Note that in the context of array
jobs, you would need to specify the array task id to rerun specific
tasks.

__Do I need to use absl-py in my project?__
No, it is not necessary for your project to use `absl-py`.
However, the launcher script uses `absl-py` to parse command line arguments.

You can use any command line argument parser you like, but just make sure
that you pass the command line arguments properly in your launch script.
