"""Behavioral contract for klygo.archive.test."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import klygo.archive as archive
import klygo.files as files

from _support import make_archive, make_source, member_ending


class TestTest(unittest.TestCase):
    def test_valid_and_corrupted_archives(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            packed = make_archive(root)
            self.assertTrue(archive.test(packed))
            broken = root / "broken.zip"
            broken.write_bytes(b"PK\x03\x04broken")
            self.assertFalse(archive.test(broken))
            with self.assertRaises(ValueError):
                archive.test(broken, raise_exception=True)


if __name__ == "__main__":
    unittest.main()

