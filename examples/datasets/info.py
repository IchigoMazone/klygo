"""Inspect a directory or archived YOLO dataset."""

from tempfile import TemporaryDirectory

from klygo import datasets, files
from _support import make_dataset


with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = make_dataset(root / "source")
    stats = datasets.info(source)
    assert stats["classes"] == ["apple", "banana"]
    assert stats["total_images"] == 4
