"""Rename classes and move class IDs."""

from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import config, datasets
from _support import make_dataset


with TemporaryDirectory() as directory:
    root = Path(directory)
    source = make_dataset(root / "source")
    output = datasets.remap_classes(
        source,
        root / "remapped",
        {"apple": "fruit", 1: 3},
        verbose=False,
    )
    metadata = config.load(output / "data.yaml", verbose=False)
    assert metadata.names == {0: "fruit", 3: "banana"}
