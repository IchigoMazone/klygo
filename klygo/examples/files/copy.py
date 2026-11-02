"""Executable example for klygo.files.copy."""

from tempfile import TemporaryDirectory
from pathlib import Path
from klygo import files

with TemporaryDirectory() as directory:
    source = Path(directory) / "source.txt"
    source.write_text("klygo", encoding="utf-8")
    copied = files.copy(source, Path(directory) / "backup" / "source.txt")
    print(copied.read_text(encoding="utf-8"))


# Additional cases
# Directories are copied recursively too.
with TemporaryDirectory() as directory:
    root = Path(directory)
    source = root / "dataset"
    source.mkdir()
    (source / "labels.txt").write_text("cat", encoding="utf-8")
    copied = files.copy(source, root / "backup")
    assert (copied / "labels.txt").read_text(encoding="utf-8") == "cat"

