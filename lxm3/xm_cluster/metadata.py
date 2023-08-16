import getpass
from typing import Set

from lxm3._vendor.xmanager.xm import metadata_context


class ClusterContextAnnotations(metadata_context.ContextAnnotations):
    def __init__(self) -> None:
        self._title = ""
        self._tags = set()
        self._notes = ""

    @property
    def title(self) -> str:
        return self._title

    def set_title(self, title: str) -> None:
        self._title = title

    @property
    def tags(self) -> Set[str]:
        return self._tags

    def add_tags(self, *tags: str) -> None:
        self._tags.update(tags)

    def remove_tags(self, *tags: str) -> None:
        for tag in tags:
            self._tags.discard(tag)

    @property
    def notes(self) -> str:
        return self._notes

    def set_notes(self, notes: str) -> None:
        self._notes = notes


class ClusterMetadataContext(metadata_context.MetadataContext):
    def __init__(self) -> None:
        super().__init__(
            creator=getpass.getuser(), annotations=ClusterContextAnnotations()
        )
