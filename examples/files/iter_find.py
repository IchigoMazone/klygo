"""Executable example for klygo.files.iter_find."""

from tempfile import TemporaryDirectory
from klygo import files

with TemporaryDirectory() as directory:
    root = files.path(directory)
    files.mkdir(root / "images")
    files.mkdir(root / "cache")
    for relative_name in ("top.jpg", "images/cat.jpg", "cache/preview.jpg"):
        (root / relative_name).touch()

    # Patterns match filenames; exclusions prune directories before scanning.
    matches = files.iter_find(root, "*.jpg", exclude_dirs=("cache",))
    assert {files.as_posix(files.relative(item, root)) for item in matches} == {"top.jpg", "images/cat.jpg"}
    assert list(files.iter_find(root, "*.jpg", recursive=False)) == [root / "top.jpg"]
