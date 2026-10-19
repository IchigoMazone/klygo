"""Behavioral contract for klygo.files.move."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestMove(unittest.TestCase):
    def test_move_and_overwrite_policy(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "a.txt", root / "nested" / "b.txt"
            source.write_text("a", encoding="utf-8")
            self.assertEqual(files.move(source, target), target)
            self.assertFalse(source.exists())
            replacement = root / "c.txt"
            replacement.write_text("c", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                files.move(replacement, target, overwrite=False)

    def test_same_directory_and_failed_copy_preserve_source(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            content = source / "keep.txt"
            content.write_text("keep", encoding="utf-8")
            for target in (source, source / "child", root):
                with self.subTest(target=target), self.assertRaises(ValueError):
                    files.move(source, target, overwrite=True)
                self.assertEqual(content.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()

