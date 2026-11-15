"""Executable example for klygo.files.copy."""

from tempfile import TemporaryDirectory
from klygo import files

with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = root / "source.txt"
    files.save(source, "klygo", verbose=False)
    copied = root / "backup" / "source.txt"
    files.copy(source, copied)
    print(files.load(copied, verbose=False))


# Additional cases
# Directories are copied recursively too.
with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = root / "dataset"
    files.mkdir(source)
    files.save(source / "labels.txt", "cat", verbose=False)
    copied = root / "backup"
    files.copy(source, copied)
    assert files.load(copied / "labels.txt", verbose=False) == "cat"
