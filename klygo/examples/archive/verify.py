"""Executable examples for klygo.archive.verify."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    packed = make_archive(Path(directory))
    report = archive.verify(packed)
    assert report["valid"] is True
    print(report)

