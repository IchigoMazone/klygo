"""Executable examples for klygo.archive.test."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = Path(directory)
    packed = make_archive(root)
    assert archive.test(packed)

    broken = root / "broken.zip"
    broken.write_bytes(b"PK\x03\x04broken")
    assert archive.test(broken) is False
    print("valid and corrupted cases checked")

