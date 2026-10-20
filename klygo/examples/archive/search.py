"""Executable examples for klygo.archive.search."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    packed = make_archive(Path(directory))
    print("glob:", archive.search(packed, "*.txt"))
    print("regex:", archive.search(packed, r".*data\.json$", regex=True))
    print("case-insensitive:", archive.search(packed, "*.TXT", case_sensitive=False))

