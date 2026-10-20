"""Executable examples for klygo.archive.split_by_size."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = Path(directory)
    packed = make_archive(root)
    parts = archive.split_by_size(packed, 0.00001, root / "parts", verbose=False)
    assert parts and all(files.is_file(part) for part in parts)
    print(parts)

