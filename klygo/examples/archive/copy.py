"""Executable examples for klygo.archive.copy."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = Path(directory)
    source = make_archive(root)
    target = root / "backup" / "dataset.zip"

    archive.copy(source, target)
    assert files.compare(source, target)
    print(target)

