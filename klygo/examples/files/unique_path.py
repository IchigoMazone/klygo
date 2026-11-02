"""Executable example for klygo.files.unique_path."""

from tempfile import TemporaryDirectory
from pathlib import Path
from klygo import files

with TemporaryDirectory() as directory:
    existing = Path(directory) / "result.json"
    existing.touch()
    print(files.unique_path(existing))


# Additional cases
# Compound extensions remain intact when numbering is needed.
with TemporaryDirectory() as directory:
    existing = Path(directory) / "dataset.tar.gz"
    existing.touch()
    assert files.unique_path(existing).name == "dataset_1.tar.gz"
    assert files.unique_path(existing, separator="-", start=4).name == "dataset-4.tar.gz"

