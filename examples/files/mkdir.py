"""Executable example for klygo.files.mkdir."""

from tempfile import TemporaryDirectory
from klygo import files

with TemporaryDirectory() as directory:
    created = files.mkdir(files.path(directory) / "outputs" / "images")
    print(created, files.is_dir(created))


# Additional cases
# Reusing the same path is safe with the default exist_ok=True.
with TemporaryDirectory() as directory:
    target = files.path(directory) / "cache"
    assert files.mkdir(target) == target
    assert files.mkdir(target) == target
