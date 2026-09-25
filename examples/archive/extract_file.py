"""Executable examples for klygo.archive.extract_file."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = Path(directory)
    packed = make_archive(root)
    names = archive.list_files(packed)
    member = member_ending(names, "data.json")

    output = root / "single"
    archive.extract_file(packed, member, output)
    assert files.is_file(output / "data.json")
    print(output / "data.json")

