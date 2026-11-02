"""Executable examples for klygo.archive.extract_matching."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = Path(directory)
    packed = make_archive(root)

    output = root / "text-members"
    archive.extract_matching(packed, "*.txt", output)
    matches = files.find(output)
    assert len(matches) == 1
    print(matches)

