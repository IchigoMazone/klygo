"""Executable examples for klygo.archive.remove."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = Path(directory)
    packed = make_archive(root)
    member = member_ending(archive.list_files(packed), "alpha.txt")

    archive.remove(packed, member)
    assert member not in archive.list_files(packed)
    print("removed:", member)

