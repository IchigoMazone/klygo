"""Executable examples for klygo.archive.compress."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = Path(directory)
    source = make_source(root)

    zip_path = root / "dataset.zip"
    archive.compress(source, zip_path, verbose=False)
    assert archive.detect_format(zip_path) == "zip"
    assert all(name.startswith("source/") for name in archive.list_files(zip_path))

    contents_only = root / "contents-only.zip"
    archive.compress(source, contents_only, include_root=False, verbose=False)
    assert "alpha.txt" in archive.list_files(contents_only)
    assert "source/alpha.txt" not in archive.list_files(contents_only)

    tar_path = root / "dataset.tar.gz"
    archive.compress(source, tar_path, compresslevel=9, verbose=False)
    assert archive.test(tar_path)

    print(zip_path, contents_only, tar_path)
