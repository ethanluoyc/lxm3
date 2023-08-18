from absl import flags


def pytest_configure(config):
    del config  # Unused.
    flags.FLAGS.mark_as_parsed()


def pytest_ignore_collect(path, config):
    if path.isfile() and path.islink():
        return True
