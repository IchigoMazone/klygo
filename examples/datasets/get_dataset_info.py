"""Inspect a directory or archived YOLO dataset."""

from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import datasets
from _support import make_dataset


with TemporaryDirectory() as directory:
    source = make_dataset(Path(directory) / "source")
    info = datasets.get_dataset_info(source)
    assert info["classes"] == ["apple", "banana"]
    assert info["total_images"] == 4
