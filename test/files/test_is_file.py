"""Behavioral contract for klygo.files.is_file."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestIsFile(unittest.TestCase):
    def test_file_directory_and_missing(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            item = root / "file.txt"
            item.touch()
            self.assertTrue(files.is_file(item))
            self.assertFalse(files.is_file(root))
            self.assertFalse(files.is_file(root / "missing"))


if __name__ == "__main__":
    unittest.main()

