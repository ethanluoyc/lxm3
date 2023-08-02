import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.seed = 1
    return config

