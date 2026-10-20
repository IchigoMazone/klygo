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
            self.assertEqual([p.name for p in files.list_entries(root, "*.txt")], ["a.txt", "B.txt"])
            self.assertEqual(len(files.list_entries(root, "*.txt", recursive=True)), 3)

    def test_rejects_missing_root(self):
        with self.assertRaises(FileNotFoundError):
            files.list_entries("definitely-missing-directory")


if __name__ == "__main__":
    unittest.main()

