"""Executable examples for klygo.archive.ArchiveFile."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = Path(directory)
    packed = make_archive(root)
    opened = archive.ArchiveFile(packed)

    print(opened.format)
    print(opened.search("*.json"))
    output = root / "restored"
    opened.extract(output, verbose=False)
    assert files.find(output)

