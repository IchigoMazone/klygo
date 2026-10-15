"""Merge datasets and reconcile local class IDs."""

from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import datasets
from _support import make_dataset


with TemporaryDirectory() as directory:
    root = Path(directory)
    first = make_dataset(root / "first", ("apple", "banana"), "first")
    second = make_dataset(root / "second", ("banana", "orange"), "second")
    output = datasets.merge([first, second], root / "merged.zip", verbose=False)
    assert datasets.get_dataset_info(output)["classes"] == ["apple", "banana", "orange"]
