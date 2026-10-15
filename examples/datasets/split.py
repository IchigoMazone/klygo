"""Create class-based and ratio-based dataset archives."""

from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import archive, datasets
from _support import make_dataset


with TemporaryDirectory() as directory:
    root = Path(directory)
    source = make_dataset(root / "source")
    outputs = datasets.split(source, root / "parts", ratios=(0.5, 0.5), verbose=False)
    assert len(outputs) == 2
    assert all(archive.is_archive(path) for path in outputs)
