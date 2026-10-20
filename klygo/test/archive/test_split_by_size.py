"""Behavioral contract for klygo.archive.split_by_size."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import klygo.archive as archive
import klygo.files as files

from _support import make_archive, make_source, member_ending


class TestSplitBySize(unittest.TestCase):
    def test_parts_are_created(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            packed = make_archive(root)
            parts = archive.split_by_size(packed, 0.00001, root / "parts", verbose=False)
            self.assertGreaterEqual(len(parts), 1)
            self.assertTrue(all(files.is_file(part) for part in parts))


if __name__ == "__main__":
    unittest.main()

