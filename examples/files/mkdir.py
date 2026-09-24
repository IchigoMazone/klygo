"""Executable example for klygo.files.mkdir."""

from tempfile import TemporaryDirectory
from pathlib import Path
from klygo import files

with TemporaryDirectory() as directory:
    created = files.mkdir(Path(directory) / "outputs" / "images")
    print(created, created.is_dir())


# Additional cases
# Reusing the same path is safe with the default exist_ok=True.
with TemporaryDirectory() as directory:
    target = Path(directory) / "cache"
    assert files.mkdir(target) == target
    assert files.mkdir(target) == target

