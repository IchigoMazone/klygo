"""Behavioral contract for klygo.files.resolve."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestResolve(unittest.TestCase):
    def test_absolute_and_strict_modes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(files.resolve(str(root)), root.resolve())
            self.assertEqual(files.resolve(root / "missing"), (root / "missing").resolve())
            with self.assertRaises(FileNotFoundError):
                files.resolve(root / "missing", strict=True)

    def test_strict_must_be_boolean(self):
        with self.assertRaises(TypeError):
            files.resolve("dataset", strict="yes")


if __name__ == "__main__":
    unittest.main()

