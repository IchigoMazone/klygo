"""Inspect optional 7-Zip support without requiring its dependency."""

from importlib.util import find_spec

from klygo.archive.backend import SevenZipBackend


backend = SevenZipBackend()
print("format:", backend.format_name)
print("can compress:", backend.capabilities.compress)
print("py7zr installed:", find_spec("py7zr") is not None)
