"""Executable example for klygo.files.move."""

from tempfile import TemporaryDirectory
from pathlib import Path
from klygo import files

with TemporaryDirectory() as directory:
    source = Path(directory) / "source.txt"
    source.write_text("klygo", encoding="utf-8")
    moved = files.move(source, Path(directory) / "output" / "result.txt")
    print(moved, source.exists())


# Additional cases
# Moving directories preserves their contents.
with TemporaryDirectory() as directory:
    root = Path(directory)
    source = root / "before"
    source.mkdir()
    (source / "data.txt").touch()
    target = files.move(source, root / "after")
    assert (target / "data.txt").is_file() and not source.exists()

