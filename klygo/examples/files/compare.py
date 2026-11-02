"""Executable example for klygo.files.compare."""

from tempfile import TemporaryDirectory
from pathlib import Path
from klygo import files

with TemporaryDirectory() as directory:
    first = Path(directory) / "a.txt"
    second = Path(directory) / "b.txt"
    first.write_text("same", encoding="utf-8")
    second.write_text("same", encoding="utf-8")
    print(files.compare(first, second, by="content"))


# Additional cases
# Different content returns False without raising.
with TemporaryDirectory() as directory:
    root = Path(directory)
    first, second = root / "a.bin", root / "b.bin"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    assert not files.compare(first, second)
    assert not files.compare(first, second, by="content")

