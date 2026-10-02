"""Behavioral contract for klygo.files.mkdir."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestMkdir(unittest.TestCase):
    def test_creates_parent_directories_and_accepts_existing_directory(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "a" / "b"
            self.assertEqual(files.mkdir(str(target)), target)
            self.assertTrue(target.is_dir())
            self.assertEqual(files.mkdir(target), target)

    def test_respects_parents_and_exist_ok_flags(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                files.mkdir(root / "missing" / "child", parents=False)
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                files.mkdir(existing, exist_ok=False)


if __name__ == "__main__":
    unittest.main()

