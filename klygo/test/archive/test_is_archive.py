"""Behavioral contract for klygo.archive.is_archive."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import klygo.archive as archive
import klygo.files as files

from _support import make_archive, make_source, member_ending


class TestIsArchive(unittest.TestCase):
    def test_supported_missing_and_plain_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            packed = make_archive(root)
            plain = root / "plain.txt"
            plain.write_text("plain", encoding="utf-8")
            self.assertTrue(archive.is_archive(packed))
            self.assertFalse(archive.is_archive(plain))
            self.assertFalse(archive.is_archive(root / "missing"))


if __name__ == "__main__":
    unittest.main()

