"""Behavioral contract for klygo.files.compare."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestCompare(unittest.TestCase):
    def test_hash_content_and_difference(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "a", root / "b"
            first.write_bytes(b"same")
            second.write_bytes(b"same")
            self.assertTrue(files.compare(first, second))
            self.assertTrue(files.compare(first, second, by="content"))
            second.write_bytes(b"different")
            self.assertFalse(files.compare(first, second))
            with self.assertRaises(ValueError):
                files.compare(first, first, by="unknown")


if __name__ == "__main__":
    unittest.main()

