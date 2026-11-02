"""Behavioral contract for klygo.archive.copy."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import klygo.archive as archive
import klygo.files as files

from _support import make_archive, make_source, member_ending


class TestCopy(unittest.TestCase):
    def test_copy_delegates_overwrite_behavior(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_archive(root)
            target = root / "backup" / "copy.zip"
            archive.copy(source, target)
            self.assertTrue(files.compare(source, target))
            with self.assertRaises(FileExistsError):
                archive.copy(source, target)


if __name__ == "__main__":
    unittest.main()

