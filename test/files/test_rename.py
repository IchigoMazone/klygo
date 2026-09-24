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


if __name__ == "__main__":
    unittest.main()

