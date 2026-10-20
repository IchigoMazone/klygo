"""Executable examples for klygo.archive.compare."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = Path(directory)
    source = make_source(root)
    first, second = root / "v1.zip", root / "v2.zip"
    archive.compress(source, first, verbose=False)
    archive.compress(source, second, verbose=False)
    extra = root / "extra.txt"
    files.save(extra, "new", verbose=False)
    archive.add(second, extra, verbose=False)

    difference = archive.compare(first, second)
    assert difference["added_files"] == ["extra.txt"]
    print(difference)

