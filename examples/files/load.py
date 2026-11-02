"""Executable example for klygo.files.load."""

from tempfile import TemporaryDirectory
from klygo import files

with TemporaryDirectory() as directory:
    source = files.path(directory) / "config.json"
    files.save(source, {"model": "yolo"}, overwrite=True, verbose=False)
    print(files.load(source, verbose=False))


# Additional cases
# Load line-oriented text without retaining newline characters.
with TemporaryDirectory() as directory:
    labels = files.path(directory) / "labels.txt"
    files.save(labels, ["cat", "dog"], verbose=False)
    assert files.load(labels, as_lines=True, verbose=False) == ["cat", "dog"]

    report = files.path(directory) / "report.md"
    files.save(report, "# Report\n\nReady", verbose=False)
    assert files.load(report, verbose=False) == "# Report\n\nReady"
