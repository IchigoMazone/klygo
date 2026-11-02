"""Behavioral contract for klygo.files.hash."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestHash(unittest.TestCase):
    def test_algorithms_and_invalid_target(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "data.bin"
            target.write_bytes(b"abc")
            self.assertEqual(files.hash(target), "900150983cd24fb0d6963f7d28e17f72")
            self.assertEqual(len(files.hash(target, "sha256")), 64)
            with self.assertRaises(ValueError):
                files.hash(directory)


if __name__ == "__main__":
    unittest.main()

