"""Behavioral contract for klygo.files.exists."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestExists(unittest.TestCase):
    def test_existing_and_missing_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            present = root / "file.txt"
            present.touch()
            self.assertTrue(files.exists(present))
            self.assertFalse(files.exists(root / "missing"))


if __name__ == "__main__":
    unittest.main()

