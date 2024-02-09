# Reference
# https://github.com/bazelbuild/bazel/blob/1c59e78215f7beaa10229df48f483779ebad8217/src/main/java/com/google/devtools/build/lib/actions/cache/DigestUtils.java

import atexit
import hashlib
import os
import pathlib
from typing import NamedTuple

import appdirs
import cachetools
import cachetools.keys
import shelved_cache

_DIGEST_CACHE = None


class _CacheKey(NamedTuple):
    path: str
    node_id: int
    time_modified: float
    size: int


def clear_cache():
    _get_cache().clear()


def _get_cache():
    global _DIGEST_CACHE
    if _DIGEST_CACHE is None:
        digest_cache_dir = pathlib.Path(appdirs.user_cache_dir("lxm3"), "digests")
        digest_cache_file = digest_cache_dir / "cache"
        digest_cache_dir.mkdir(parents=True, exist_ok=True)
        _DIGEST_CACHE = shelved_cache.PersistentCache(
            cachetools.LRUCache, filename=str(digest_cache_file), maxsize=1000
        )
        atexit.register(_DIGEST_CACHE.close)
    return _DIGEST_CACHE


def sha256_digest(filename: str):
    path = os.path.abspath(filename)
    stat = os.stat(path)

    cache_key = cachetools.keys.hashkey(
        _CacheKey(path, stat.st_ino, stat.st_mtime, stat.st_size)
    )
    cache = _get_cache()
    if cache_key in cache:
        return cache[cache_key]
    else:
        digest = f"sha256:{_sha256sum(path)}"
        cache[cache_key] = digest
        return digest


def _sha256sum(path: str, block_size=2**20):
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(block_size)
            if not data:
                break
            sha256.update(data)
        return sha256.hexdigest()


if __name__ == "__main__":
    print(sha256_digest(__file__))
