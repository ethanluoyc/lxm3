import os

from absl import app
from absl import flags

_SEED = flags.DEFINE_integer("seed", 0, "Random seed")


def main(_):
    print("--seed", _SEED.value, "TASK", os.environ["TASK"])


if __name__ == "__main__":
    app.run(main)
