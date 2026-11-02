"""Executable example for klygo.files.rename."""

from tempfile import TemporaryDirectory
from pathlib import Path
from klygo import files

with TemporaryDirectory() as directory:
    source = Path(directory) / "draft.txt"
    source.write_text("done", encoding="utf-8")
    renamed = files.rename(source, "final.txt")
    print(renamed)


# Additional cases
# A full destination path can move and rename in one operation.
with TemporaryDirectory() as directory:
    root = Path(directory)
    source = root / "a.txt"
    source.touch()
    destination = root / "nested" / "b.txt"
    destination.parent.mkdir()
    assert files.rename(source, destination) == destination

