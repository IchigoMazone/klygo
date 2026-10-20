"""Behavioral contract for klygo.archive.compare."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import klygo.archive as archive
import klygo.files as files

from _support import make_archive, make_source, member_ending


class TestCompare(unittest.TestCase):
    def test_added_removed_and_common_members(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_source(root, "shared")
            first = root / "first.zip"
            second = root / "second.zip"
            archive.compress(source, first, verbose=False)
            archive.compress(source, second, verbose=False)
            extra = root / "extra.txt"
            extra.write_text("x", encoding="utf-8")
            archive.add(second, extra, verbose=False)
            result = archive.compare(first, second)
            self.assertIn("extra.txt", result["added_files"])
            self.assertTrue(result["common_files"])


if __name__ == "__main__":
    unittest.main()
