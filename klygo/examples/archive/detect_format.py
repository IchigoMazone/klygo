"""Executable examples for klygo.archive.detect_format."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


assert archive.detect_format("dataset.zip") == "zip"
assert archive.detect_format("dataset.tar.gz") == "tar.gz"
assert archive.detect_format("dataset.tbz2") == "tar.bz2"

with TemporaryDirectory() as directory:
    root = Path(directory)
    packed = make_archive(root)
    renamed = files.move(packed, root / "archive.bin")
    assert archive.detect_format(renamed) == "zip"
    print("magic-byte detection:", renamed)

