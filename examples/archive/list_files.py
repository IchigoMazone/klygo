"""Executable examples for klygo.archive.list_files."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    packed = make_archive(Path(directory))
    names = archive.list_files(packed)
    assert any(name.endswith("alpha.txt") for name in names)
    assert any(name.endswith("data.json") for name in names)
    print(*names, sep="\n")

