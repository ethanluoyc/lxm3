import os

from absl import flags
from absl import app
from ml_collections import config_flags

from py_package import lib

_CONFIG = config_flags.DEFINE_config_file("config", None)
flags.mark_flag_as_required("config")
flags.DEFINE_integer("task", 0, "Task ID")


def main(_):
    print("config:\n", _CONFIG.value)
    print("task", flags.FLAGS.task, "SGE_TASK_ID", os.environ.get("SGE_TASK_ID"))
    print(f"1 + 1 = {lib.add(1, 1)}")


if __name__ == "__main__":
    app.run(main)
