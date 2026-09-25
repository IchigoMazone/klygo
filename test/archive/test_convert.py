"""Behavioral contract for klygo.archive.convert."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import klygo.archive as archive
import klygo.files as files

from _support import make_archive, make_source, member_ending


class TestConvert(unittest.TestCase):
    def test_zip_to_tar_gz(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_archive(root)
            target = root / "converted.tar.gz"
            archive.convert(source, target, verbose=False)
            self.assertEqual(archive.detect_format(target), "tar.gz")
            self.assertEqual(
                set(archive.list_files(target)),
                set(archive.list_files(source)),
            )


if __name__ == "__main__":
    unittest.main()
