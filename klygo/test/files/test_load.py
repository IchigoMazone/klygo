"""Behavioral contract for klygo.files.load."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestLoad(unittest.TestCase):
    def test_supported_inputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            files.save(root / "data.json", {"a": 1}, verbose=False)
            files.save(root / "lines.txt", ["a", "b"], verbose=False)
            self.assertEqual(files.load(root / "data.json", verbose=False), {"a": 1})
            self.assertEqual(files.load(root / "lines.txt", as_lines=True, verbose=False), ["a", "b"])

    def test_errors(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                files.load(root / "missing.json", verbose=False)
            unsupported = root / "data.bin"
            unsupported.touch()
            with self.assertRaises(ValueError):
                files.load(unsupported, verbose=False)


if __name__ == "__main__":
    unittest.main()

