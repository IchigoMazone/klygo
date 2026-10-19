"""Executable example for klygo.files.rename."""

from tempfile import TemporaryDirectory
from klygo import files

with TemporaryDirectory() as directory:
    source = files.path(directory) / "draft.txt"
    files.save(source, "done", verbose=False)
    renamed = files.rename(source, "final.txt")
    print(renamed)


# Additional cases
# A full destination path can move and rename in one operation.
with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = root / "a.txt"
    files.save(source, "item", verbose=False)
    destination = root / "nested" / "b.txt"
    files.mkdir(files.parent(destination))
    assert files.rename(source, destination) == destination
