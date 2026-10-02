"""Behavioral contract for klygo.files.move."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestMove(unittest.TestCase):
    def test_moves_file_creates_parent_and_overwrites_target(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "a.txt", root / "nested" / "b.txt"
            source.write_text("a", encoding="utf-8")
            self.assertEqual(files.move(str(source), target), target)
            self.assertFalse(source.exists())
            self.assertEqual(target.read_text(encoding="utf-8"), "a")
            replacement = root / "c.txt"
            replacement.write_text("c", encoding="utf-8")
            self.assertEqual(files.move(replacement, target), target)
            self.assertEqual(target.read_text(encoding="utf-8"), "c")

    def test_rejects_missing_source_and_existing_target_without_overwrite(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            target = root / "target.txt"
            source.write_text("source", encoding="utf-8")
            target.write_text("target", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                files.move(source, target, overwrite=False)
            with self.assertRaises(FileNotFoundError):
                files.move(root / "missing.txt", root / "new.txt")


if __name__ == "__main__":
    unittest.main()

