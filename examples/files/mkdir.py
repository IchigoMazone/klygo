"""Executable example for klygo.files.mkdir."""

from tempfile import TemporaryDirectory
from klygo import files

with TemporaryDirectory() as directory:
    created = files.path(directory) / "outputs" / "images"
    files.mkdir(created)
    print(created, files.is_dir(created))


# Additional cases
# Reusing the same path is safe with the default exist_ok=True.
with TemporaryDirectory() as directory:
    target = files.path(directory) / "cache"
    assert files.mkdir(target) is None
    assert files.mkdir(target) is None
    assert files.is_dir(target)
