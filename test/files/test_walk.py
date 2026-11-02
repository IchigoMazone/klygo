"""Behavioral contract for klygo.files.walk."""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestWalk(unittest.TestCase):
    def test_walk_is_lazy_and_complete(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "nested" / "a.txt").touch()
            result = files.walk(root)
            self.assertTrue(hasattr(result, "__next__"))
            rows = list(result)
            self.assertEqual(len(rows), 2)
            self.assertIn("a.txt", rows[1][2])

    def test_string_root_spelling_and_empty_root_are_preserved(self):
        with TemporaryDirectory() as directory:
            root = directory + os.sep
            self.assertEqual(next(files.walk(root))[0], root)
        self.assertEqual(list(files.walk("")), [])


if __name__ == "__main__":
    unittest.main()

