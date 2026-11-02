"""Executable examples for klygo.archive.open."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    packed = make_archive(Path(directory))
    with archive.open(packed) as opened:
        print("format:", opened.format)
        print("members:", opened.list_files())
        assert opened.test()

