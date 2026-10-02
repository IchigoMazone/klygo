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
            (root / "archive_1.tar.gz").touch()
            self.assertEqual(files.unique_path(str(existing)), root / "archive_2.tar.gz")

    def test_validates_numbering_arguments(self):
        with TemporaryDirectory() as directory:
            existing = Path(directory) / "result.txt"
            existing.touch()
            self.assertEqual(files.unique_path(existing, separator="-", start=0), Path(directory) / "result-0.txt")
            with self.assertRaises(ValueError):
                files.unique_path(existing, start=-1)
            with self.assertRaises(TypeError):
                files.unique_path(existing, separator=1)


if __name__ == "__main__":
    unittest.main()

