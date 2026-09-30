"""Behavioral contract for klygo.files.copy."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestCopy(unittest.TestCase):
    def test_file_directory_and_overwrite_policy(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            target = root / "nested" / "target.txt"
            source.write_text("one", encoding="utf-8")
            self.assertEqual(files.copy(source, target).read_text(encoding="utf-8"), "one")
            with self.assertRaises(FileExistsError):
                files.copy(source, target, overwrite=False)
            folder = root / "folder"
            folder.mkdir()
            (folder / "a.txt").touch()
            copied = files.copy(folder, root / "folder-copy")
            self.assertTrue((copied / "a.txt").is_file())

    def test_overwrite_replaces_existing_file_and_directory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source.txt", root / "target.txt"
            source.write_text("new", encoding="utf-8")
            target.write_text("old", encoding="utf-8")
            self.assertEqual(files.copy(source, target), target)
            self.assertEqual(target.read_text(encoding="utf-8"), "new")

            source_dir, target_dir = root / "source", root / "target"
            source_dir.mkdir()
            (source_dir / "current.txt").write_text("current", encoding="utf-8")
            target_dir.mkdir()
            (target_dir / "stale.txt").write_text("stale", encoding="utf-8")
            files.copy(source_dir, target_dir)
            self.assertTrue((target_dir / "current.txt").is_file())
            self.assertFalse((target_dir / "stale.txt").exists())

    def test_missing_source_raises_file_not_found(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                files.copy(root / "missing.txt", root / "target.txt")


if __name__ == "__main__":
    unittest.main()

