"""Executable examples for klygo.archive.add."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = Path(directory)
    packed = make_archive(root)
    extra = root / "extra.txt"
    files.save(extra, "extra", verbose=False)

    archive.add(packed, extra, verbose=False)
    archive.add(packed, extra, on_conflict="rename", verbose=False)
    names = archive.list_files(packed)
    assert "extra.txt" in names
    assert any("extra_dup" in name for name in names)
    print(names)

