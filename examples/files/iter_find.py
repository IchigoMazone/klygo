"""Executable example for klygo.files.iter_find."""

from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files

with TemporaryDirectory() as directory:
    root = Path(directory)
    files.mkdir(root / "images")
    files.mkdir(root / "cache")
    for name in ("top.jpg", "images/cat.jpg", "cache/preview.jpg"):
        (root / name).touch()

    # Patterns match filenames; exclusions prune directories before scanning.
    matches = files.iter_find(root, "*.jpg", exclude_dirs=("cache",))
    assert {item.relative_to(root).as_posix() for item in matches} == {"top.jpg", "images/cat.jpg"}
    assert list(files.iter_find(root, "*.jpg", recursive=False)) == [root / "top.jpg"]
