"""Behavioral contract for klygo.files.list_entries."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestListEntries(unittest.TestCase):
    def test_patterns_sorting_and_recursion(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "B.txt").touch()
            (root / "a.txt").touch()
            (root / "nested" / "c.txt").touch()
            self.assertEqual(
                [entry.name for entry in files.list_entries(str(root))],
                ["a.txt", "B.txt", "nested"],
            )
            self.assertEqual(
                [entry.name for entry in files.list_entries(root, "*.txt", recursive=True)],
                ["a.txt", "B.txt", "c.txt"],
            )

    def test_rejects_missing_and_file_roots(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            file_path = root / "file.txt"
            file_path.touch()
            with self.assertRaises(FileNotFoundError):
                files.list_entries(root / "missing")
            with self.assertRaises(ValueError):
                files.list_entries(file_path)


if __name__ == "__main__":
    unittest.main()

