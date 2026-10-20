"""Behavioral contract for klygo.files.is_dir."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestIsDir(unittest.TestCase):
    def test_directory_file_and_missing(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            item = root / "file.txt"
            item.touch()
            self.assertTrue(files.is_dir(root))
            self.assertFalse(files.is_dir(item))
            self.assertFalse(files.is_dir(root / "missing"))


if __name__ == "__main__":
    unittest.main()

