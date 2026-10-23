"""Executable example for klygo.files.copy."""

from tempfile import TemporaryDirectory
from klygo import files

with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = root / "source.txt"
    files.save(source, "klygo", verbose=False)
    copied = files.copy(source, root / "backup" / "source.txt")
    print(files.load(copied, verbose=False))


# Additional cases
# Directories are copied recursively too.
with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = files.mkdir(root / "dataset")
    files.save(source / "labels.txt", "cat", verbose=False)
    copied = files.copy(source, root / "backup")
    assert files.load(copied / "labels.txt", verbose=False) == "cat"
