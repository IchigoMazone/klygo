"""Executable example for klygo.files.move."""

from tempfile import TemporaryDirectory
from klygo import files

with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = root / "source.txt"
    files.save(source, "klygo", verbose=False)
    moved = files.move(source, root / "output" / "result.txt")
    print(moved, files.exists(source))


# Additional cases
# Moving directories preserves their contents.
with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = files.mkdir(root / "before")
    files.save(source / "data.txt", "ready", verbose=False)
    target = files.move(source, root / "after")
    assert files.is_file(target / "data.txt") and not files.exists(source)
