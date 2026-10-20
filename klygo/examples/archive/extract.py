"""Executable examples for klygo.archive.extract."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = Path(directory)
    packed = make_archive(root)

    all_output = root / "all"
    archive.extract(packed, all_output, verbose=False)
    print(files.find(all_output))

    json_output = root / "json-only"
    archive.extract(packed, json_output, include="*.json", verbose=False)
    assert all(path.suffix == ".json" for path in files.find(json_output))

