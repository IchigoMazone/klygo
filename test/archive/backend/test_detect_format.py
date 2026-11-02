"""Behavioral tests for backend-level detect_format."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo.archive.backend import detect_format


class TestDetectFormat(unittest.TestCase):
    def test_compound_extensions_and_aliases(self):
        self.assertEqual(detect_format("data.tar.gz"), "tar.gz")
        self.assertEqual(detect_format("data.tgz"), "tar.gz")
        self.assertEqual(detect_format("data.tbz2"), "tar.bz2")

    def test_magic_bytes_override_unknown_suffix(self):
        with TemporaryDirectory() as directory:
            candidate = Path(directory) / "archive.bin"
            candidate.write_bytes(b"PK\x03\x04" + b"\0" * 20)
            self.assertEqual(detect_format(candidate), "zip")

    def test_unknown_format_raises(self):
        with self.assertRaises(ValueError):
            detect_format("notes.txt")


if __name__ == "__main__":
    unittest.main()

