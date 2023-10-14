# type: ignore

from absl import app

from lxm3 import xm
from lxm3 import xm_cluster


def main(_):
    with xm_cluster.create_experiment(experiment_title="universal") as experiment:
        executor = xm_cluster.Local()
        executable = xm.Packageable(
            xm_cluster.UniversalPackage(
                path=".",
                entrypoint=["python3", "main.py"],
                build_script="build.sh",
            ),
            xm_cluster.Local.Spec(),  # type: ignore
        )
        [executable] = experiment.package([executable])
        experiment.add(xm.Job(executable=executable, executor=executor))


if __name__ == "__main__":
    app.run(main)
