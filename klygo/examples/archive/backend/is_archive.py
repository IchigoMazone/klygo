"""Probe archive-like names without handling detection exceptions."""

from klygo.archive.backend import is_archive


for filename in ("dataset.zip", "labels.tar.gz", "README.md"):
    print(filename, is_archive(filename))

