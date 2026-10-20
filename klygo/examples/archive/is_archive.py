"""Executable examples for klygo.archive.is_archive."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = Path(directory)
    packed = make_archive(root)
    plain = root / "notes.txt"
    files.save(plain, "plain text", verbose=False)

    assert archive.is_archive(packed)
    assert not archive.is_archive(plain)
    assert not archive.is_archive(root / "missing")
    print("archive detection checked")

