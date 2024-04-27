from absl import flags


def pytest_configure(config):
    del config  # Unused.
    flags.FLAGS.mark_as_parsed()


def pytest_ignore_collect(collection_path, config):
    if collection_path.is_file() and collection_path.is_symlink():
        return True
