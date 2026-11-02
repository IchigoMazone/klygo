"""Executable examples for klygo.archive.iter_files."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    packed = make_archive(Path(directory))
    iterator = archive.iter_files(packed)
    assert hasattr(iterator, "__next__")
    for member in iterator:
        print(member)

