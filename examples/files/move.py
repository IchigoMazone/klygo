"""Executable example for klygo.files.move."""

from tempfile import TemporaryDirectory
from klygo import files

with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = root / "source.txt"
    files.save(source, "klygo", verbose=False)
    moved = root / "output" / "result.txt"
    files.move(source, moved)
    print(moved, files.exists(source))


# Additional cases
# Moving directories preserves their contents.
with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = root / "before"
    files.mkdir(source)
    files.save(source / "data.txt", "ready", verbose=False)
    target = root / "after"
    files.move(source, target)
    assert files.is_file(target / "data.txt") and not files.exists(source)
