"""Flatten all dataset splits into one image and label directory."""

from tempfile import TemporaryDirectory

from klygo import datasets, files
from _support import make_dataset


with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = make_dataset(root / "source")
    output = datasets.unpartition(source, root / "flat", verbose=False)
    assert datasets.info(output)["splits"] == {"raw": 4}
