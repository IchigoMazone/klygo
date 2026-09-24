"""Executable example for klygo.files.save."""

from tempfile import TemporaryDirectory
from pathlib import Path
from klygo import files

with TemporaryDirectory() as directory:
    output = Path(directory) / "config.yaml"
    files.save(output, {"batch": 16}, overwrite=True, verbose=False)
    print(output.read_text(encoding="utf-8"))


# Additional cases
# The extension selects the serializer; parent folders are created.
with TemporaryDirectory() as directory:
    root = Path(directory)
    files.save(root / "nested" / "records.jsonl", [{"id": 1}, {"id": 2}], verbose=False)
    assert files.load(root / "nested" / "records.jsonl", verbose=False) == [{"id": 1}, {"id": 2}]

