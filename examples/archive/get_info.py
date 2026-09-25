"""Executable examples for klygo.archive.get_info."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    packed = make_archive(Path(directory))
    metadata = archive.get_info(packed)
    print("format:", metadata["format"])
    print("members:", metadata["file_count"])
    print("archive size:", metadata["human_archive_size"])
    assert metadata["format"] == "zip"

