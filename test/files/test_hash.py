"""Behavioral contract for klygo.files.hash."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestHash(unittest.TestCase):
    def test_algorithms_and_string_path(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "data.bin"
            target.write_bytes(b"abc")
            self.assertEqual(files.hash(str(target)), "900150983cd24fb0d6963f7d28e17f72")
            self.assertEqual(
                files.hash(target, "sha256"),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )

    def test_missing_file_and_directory_are_rejected(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                files.hash(directory)
            with self.assertRaises(FileNotFoundError):
                files.hash(Path(directory) / "missing.bin")


if __name__ == "__main__":
    unittest.main()

