"""Executable examples for klygo.archive.merge."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = Path(directory)
    source = make_source(root)
    first, second = root / "a.zip", root / "b.zip"
    archive.compress(source, first, verbose=False)
    archive.compress(source, second, verbose=False)

    merged = root / "merged.zip"
    archive.merge([first, second], merged, verbose=False)
    assert archive.test(merged)
    print(archive.list_files(merged))

