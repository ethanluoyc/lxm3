# Tips and Tricks

## 1. Set up WandB logging for your project

You can configure environment variables per work unit to automatically
generate log each job to a [Weights and Biases](https://wandb.ai/site)
run with name and group.

```python
def _get_wandb_env_vars(work_unit: xm.WorkUnit, experiment_name: str):
    xid = work_unit.experiment_id
    wid = work_unit.work_unit_id
    env_vars = {
        "WANDB_PROJECT": _WANDB_PROJECT.value,
        "WANDB_ENTITY": _WANDB_ENTITY.value,
        "WANDB_NAME": f"{experiment_name}_{xid}_{wid}",
        "WANDB_MODE": _WANDB_MODE.value,
    }
    if _WANDB_GROUP.value is not None:
        env_vars.update(
            {
                "WANDB_RUN_GROUP": _WANDB_GROUP.value.format(
                    name=experiment_name, xid=xid
                )
            }
        )
    try:
        import git

        commit_sha = git.Repo().head.commit.hexsha
        env_vars["WANDB_GIT_REMOTE_URL"] = git.Repo().remote().url
        env_vars["WANDB_GIT_COMMIT"] = commit_sha
    except Exception:
        logging.info("Unable to parse git info.")
    return env_vars

# Then when you create jobs pass these to your job
experiment.add(
    xm.Job(
        executable, executor,
        args=args,
        env_vars=_get_wandb_env_vars(work_unit, exp_name))
)

```

## 2. Use `ml_collections` for your hyperparameters

If you use
[`ml_collections.ConfigDict`](https://github.com/google/ml_collections/tree/master)
to manage your hyperparameters, you can pass the config file to your
job while allowing overriding specific configuration from the launcher
like the following:
```python
# Define a config flag in your launcher
config_flags.DEFINE_config_file("config", None, "Path to config")
FLAGS = flags.FLAGS

# Create a Fileset resource to pass to your PythonPackage
config_resource = xm_cluster.Fileset(
    # FLAGS["config] accesses the actual flag.
    files={config_flags.get_config_filename(FLAGS["config"]): "config.py"}
)

spec = xm_cluster.PythonPackage(
    entrypoint=xm_cluster.ModuleName(_ENTRYPOINT.value),
    path=".",
    resources=[config_resource],
)

# Resolve a path to the config resource and pass to your executable
args = {"config": config_resource.get_path("config.py", executor.Spec())}
# Find out overrides from the command line, forward those too.
overrides = config_flags.get_override_values(FLAGS["config"])
# Prefix the `config.` as the override values do not contain them.
overrides = {f"config.{k}": v for k, v in overrides.items()}
print(f"Overrides: {overrides}")
args.update(overrides)
[executable] = experiment.package(
    [xm.Packageable(spec, executor.Spec(), args=args)]
)

```
