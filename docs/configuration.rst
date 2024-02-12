Configuration
=================

Configuration file
##################

LXM3 uses a TOML configuration file to store global configuration information.

The following places are checked in order for LXM3 configuration:
    1. Path to a configuration file specified via ``--lxm_config=<path to config>``.
    2. Path to a configuration file specified by the ``LXM_CONFIG`` environment variable.
    3. lxm.toml file in the current working directory.
    4. Configuration file stored in ``$XDG_CONFIG_HOME/lxm3/config.toml`` (typically ``~/.config/lxm3/config.toml``)

Here is an example Configuration file:

.. literalinclude:: lxm.toml
    :language: toml


Environment variables
#####################

The following environment variables can be used to configure the LXM3
launcher:

1. ``LXM_CONFIG``: Path to the configuration file.
2. ``LXM_PROJECT``: Name of the project to use. This is used as
    the sub-directory used when staging jobs.
3. ``LXM_CLUSTER``: Name of default cluster to use for launching jobs.
