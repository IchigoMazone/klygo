"""Behavioral contract for klygo.files.walk."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestWalk(unittest.TestCase):
    def test_walk_is_lazy_and_complete(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "root.txt").touch()
            (root / "nested" / "a.txt").touch()
            result = files.walk(str(root))
            self.assertTrue(hasattr(result, "__next__"))
            rows = list(result)
            self.assertEqual(len(rows), 2)
            entries = {Path(current): (directories, filenames) for current, directories, filenames in rows}
            self.assertEqual(entries[root][0], ["nested"])
            self.assertEqual(entries[root][1], ["root.txt"])
            self.assertEqual(entries[root / "nested"][1], ["a.txt"])


if __name__ == "__main__":
    unittest.main()

