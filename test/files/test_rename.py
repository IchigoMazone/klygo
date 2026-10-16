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
            source.touch()
            renamed = files.rename(source, "final.txt")
            self.assertEqual(renamed, root / "final.txt")
            self.assertTrue(renamed.exists())
            with self.assertRaises(FileExistsError):
                files.rename(renamed, renamed)


    def test_overwrite_replaces_existing_file(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "new.txt"
            target = root / "old.txt"
            source.write_text("new", encoding="utf-8")
            target.write_text("old", encoding="utf-8")
            self.assertEqual(files.rename(source, target, overwrite=True), target)
            self.assertEqual(target.read_text(encoding="utf-8"), "new")
            self.assertFalse(source.exists())


if __name__ == "__main__":
    unittest.main()

