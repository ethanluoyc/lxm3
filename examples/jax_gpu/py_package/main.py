import jax  # noqa
from absl import app


def main(_):
    print(jax.devices())


if __name__ == "__main__":
    app.run(main)
