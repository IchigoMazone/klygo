"""Executable example for klygo.files.compare."""

from tempfile import TemporaryDirectory
from klygo import files

with TemporaryDirectory() as directory:
    root = files.path(directory)
    first = root / "a.txt"
    second = root / "b.txt"
    files.save(first, "same", verbose=False)
    files.save(second, "same", verbose=False)
    print(files.compare(first, second, by="content"))


# Additional cases
# Different content returns False without raising.
with TemporaryDirectory() as directory:
    root = files.path(directory)
    first, second = root / "a.txt", root / "b.txt"
    files.save(first, "one", verbose=False)
    files.save(second, "two", verbose=False)
    assert not files.compare(first, second)
    assert not files.compare(first, second, by="content")
