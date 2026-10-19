"""Executable examples for klygo.archive.recompress."""

from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = make_archive(root)
    target = root / "compact.zip"

    archive.recompress(source, target, compresslevel=9, verbose=False)
    assert archive.test(target)
    print(files.size(source), files.size(target))
