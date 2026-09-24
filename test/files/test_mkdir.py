"""Behavioral contract for klygo.files.mkdir."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestMkdir(unittest.TestCase):
    def test_parents_and_existing_directory(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "a" / "b"
            self.assertEqual(files.mkdir(target), target)
            self.assertTrue(target.is_dir())
            self.assertEqual(files.mkdir(target), target)


if __name__ == "__main__":
    unittest.main()

