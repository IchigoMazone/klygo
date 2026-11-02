"""Behavioral contract for klygo.files.compare."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from klygo import files


class TestCompare(unittest.TestCase):
    def test_hash_content_and_difference(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "a", root / "b"
            first.write_bytes(b"same")
            second.write_bytes(b"same")
            with patch("klygo.files.metadata.hash", side_effect=AssertionError("unexpected hash")):
                self.assertTrue(files.compare(first, second))
            self.assertTrue(files.compare(first, second, by="content"))
            self.assertTrue(files.compare(first, second, by="hash"))
            second.write_bytes(b"different")
            self.assertFalse(files.compare(first, second))
            with self.assertRaises(ValueError):
                files.compare(first, first, by="unknown")
            with self.assertRaises(ValueError):
                files.compare(first, second, by="unknown")


if __name__ == "__main__":
    unittest.main()

