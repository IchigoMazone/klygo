"""Behavioral contract for klygo.files.rename."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestRename(unittest.TestCase):
    def test_name_and_explicit_path(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "draft.txt"
            source.write_text("draft", encoding="utf-8")
            renamed = files.rename(str(source), "final.txt")
            self.assertEqual(renamed, root / "final.txt")
            self.assertEqual(renamed.read_text(encoding="utf-8"), "draft")
            moved = files.rename(renamed, root / "archive.txt")
            self.assertEqual(moved, root / "archive.txt")

    def test_rejects_existing_target_and_missing_source(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.touch()
            target = root / "target.txt"
            target.touch()
            with self.assertRaises(FileExistsError):
                files.rename(source, target)
            with self.assertRaises(FileNotFoundError):
                files.rename(root / "missing.txt", "new.txt")


if __name__ == "__main__":
    unittest.main()

