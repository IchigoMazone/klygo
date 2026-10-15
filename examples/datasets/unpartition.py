"""Flatten all dataset splits into one image and label directory."""

from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import datasets
from _support import make_dataset


with TemporaryDirectory() as directory:
    root = Path(directory)
    source = make_dataset(root / "source")
    output = datasets.unpartition(source, root / "flat", verbose=False)
    assert datasets.get_dataset_info(output)["splits"] == {"raw": 4}
