"""Executable example for klygo.files.load."""

from tempfile import TemporaryDirectory
from pathlib import Path
from klygo import files

with TemporaryDirectory() as directory:
    source = Path(directory) / "config.json"
    files.save(source, {"model": "yolo"}, overwrite=True, verbose=False)
    print(files.load(source, verbose=False))


# Additional cases
# Load line-oriented text without retaining newline characters.
with TemporaryDirectory() as directory:
    labels = Path(directory) / "labels.txt"
    labels.write_text("cat\ndog\n", encoding="utf-8")
    assert files.load(labels, as_lines=True, verbose=False) == ["cat", "dog"]

