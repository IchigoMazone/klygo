"""Behavioral contract for klygo.files.remove."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestRemove(unittest.TestCase):
    def test_removes_file_and_directory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            item = root / "item.txt"
            item.touch()
            self.assertIsNone(files.remove(str(item)))
            self.assertFalse(item.exists())
            folder = root / "folder"
            folder.mkdir()
            (folder / "child").touch()
            files.remove(folder)
            self.assertFalse(folder.exists())

    def test_missing_path_policy(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(files.remove(root / "missing"))
            with self.assertRaises(FileNotFoundError):
                files.remove(root / "missing", missing_ok=False)


if __name__ == "__main__":
    unittest.main()

