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


if __name__ == "__main__":
    unittest.main()

