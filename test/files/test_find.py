"""Behavioral contract for klygo.files.find."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestFind(unittest.TestCase):
    def test_returns_only_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "a.py").touch()
            (root / "nested" / "b.py").touch()
            self.assertEqual(len(files.find(root, "*.py")), 2)
            self.assertEqual(len(files.find(root, "*.py", recursive=False)), 1)


if __name__ == "__main__":
    unittest.main()

