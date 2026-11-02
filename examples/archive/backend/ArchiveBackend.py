"""Inspect the abstract contract required from a custom backend."""

from klygo.archive.backend import ArchiveBackend


required = sorted(ArchiveBackend.__abstractmethods__)
print("required read methods:", required)
print("optional mutation defaults:", ["compress", "add", "remove", "merge", "split_by_size"])

