"""Executable examples for klygo.archive.convert."""

from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = make_archive(root)
    target = root / "converted.tar.gz"

    archive.convert(source, target, verbose=False)
    assert archive.detect_format(target) == "tar.gz"
    assert archive.test(target)
    print(target)
