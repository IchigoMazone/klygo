"""Behavioral contract for klygo.files.unique_path."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestUniquePath(unittest.TestCase):
    def test_available_numbered_and_compound_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            available = root / "free.txt"
            self.assertEqual(files.unique_path(available), available)
            existing = root / "archive.tar.gz"
            existing.touch()
            self.assertEqual(files.unique_path(existing), root / "archive_1.tar.gz")
            with self.assertRaises(ValueError):
                files.unique_path(existing, start=-1)


if __name__ == "__main__":
    unittest.main()

