"""Behavioral contract for klygo.archive.detect_format."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import klygo.archive as archive
import klygo.files as files

from _support import make_archive, make_source, member_ending


class TestDetectFormat(unittest.TestCase):
    def test_extensions_magic_and_unknown(self):
        self.assertEqual(archive.detect_format("data.tar.gz"), "tar.gz")
        self.assertEqual(archive.detect_format("data.tbz2"), "tar.bz2")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            packed = make_archive(root, "extensionless.zip")
            renamed = root / "archive.bin"
            files.move(packed, renamed)
            self.assertEqual(archive.detect_format(renamed), "zip")
            with self.assertRaises(ValueError):
                archive.detect_format(root / "unknown.bin")


if __name__ == "__main__":
    unittest.main()

