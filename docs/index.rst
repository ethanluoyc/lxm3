.. lxm3 documentation master file, created by
   sphinx-quickstart on Sat Oct 14 19:55:49 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LXM3's documentation!
================================

LXM3 provides an implementation for DeepMind's
`XManager <https://github.com/deepmind/xmanager/tree/main>`_ launch API
that aims to provide a similar experience for running experiments on
traditional HPC (SGE, Slurm).

With LXM3, launching your job on a HPC cluster is as simple as
creating a launch script, e.g. ``launcher.py``:

.. code-block:: python

   with xm_cluster.create_experiment(experiment_title="Hello World") as experiment:
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


and run the command

.. code-block::

   lxm3 launch launcher.py

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   configuration
   tips_and_tricks
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
