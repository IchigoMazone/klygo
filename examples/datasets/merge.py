"""Merge datasets and reconcile local class IDs."""

from tempfile import TemporaryDirectory

from klygo import datasets, files
from _support import make_dataset


with TemporaryDirectory() as directory:
    root = files.path(directory)
    first = make_dataset(root / "first", ("apple", "banana"), "first")
    second = make_dataset(root / "second", ("banana", "orange"), "second")
    output = root / "merged.zip"
    datasets.merge([first, second], output, verbose=False)
    assert datasets.info(output)["classes"] == ["apple", "banana", "orange"]
