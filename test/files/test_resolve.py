"""Behavioral contract for klygo.files.resolve."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestResolve(unittest.TestCase):
    def test_absolute_and_strict_modes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertTrue(files.resolve(root).is_absolute())
            with self.assertRaises(FileNotFoundError):
                files.resolve(root / "missing", strict=True)


if __name__ == "__main__":
    unittest.main()

