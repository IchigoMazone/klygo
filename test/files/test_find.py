"""Behavioral contract for klygo.files.find."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestFind(unittest.TestCase):
    def test_filters_files_recursively_and_sorts_results(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "nested" / "ignored.py").mkdir()
            first = root / "a.py"
            second = root / "nested" / "b.py"
            first.touch()
            second.touch()
            self.assertEqual(files.find(root, "*.py"), [first, second])
            self.assertEqual(files.find(str(root), "*.py", recursive=False), [first])

    def test_missing_root_raises_file_not_found_error(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                files.find(Path(directory) / "missing")


if __name__ == "__main__":
    unittest.main()

