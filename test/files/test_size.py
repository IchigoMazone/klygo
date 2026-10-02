"""Behavioral contract for klygo.files.size."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestSize(unittest.TestCase):
    def test_file_directory_and_human_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.bin").write_bytes(b"123")
            (root / "b.bin").write_bytes(b"45")
            self.assertEqual(files.size(str(root / "a.bin")), 3)
            self.assertEqual(files.size(root), 5)
            self.assertEqual(files.size(root, human=True), "5.00 B")

    def test_missing_path_raises_file_not_found_error(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                files.size(Path(directory) / "missing")


if __name__ == "__main__":
    unittest.main()

