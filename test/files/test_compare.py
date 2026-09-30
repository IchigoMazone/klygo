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

    def test_empty_files_and_string_paths_are_equal(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "a", root / "b"
            first.touch()
            second.touch()
            self.assertTrue(files.compare(str(first), str(second)))
            self.assertTrue(files.compare(str(first), str(second), by="content"))

    def test_same_size_different_content_is_not_equal(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "a", root / "b"
            first.write_bytes(b"abc")
            second.write_bytes(b"xyz")
            self.assertFalse(files.compare(first, second))
            self.assertFalse(files.compare(first, second, by="content"))

    def test_content_comparison_spans_multiple_chunks(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "a", root / "b"
            content = b"a" * 65536 + b"same tail"
            first.write_bytes(content)
            second.write_bytes(content)
            self.assertTrue(files.compare(first, second, by="content"))
            second.write_bytes(b"a" * 65536 + b"other tail")
            self.assertFalse(files.compare(first, second, by="content"))

    def test_missing_file_raises_file_not_found(self):
        with TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            with self.assertRaises(FileNotFoundError):
                files.compare(missing, missing)

    def test_invalid_mode_raises_before_size_comparison(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "a", root / "b"
            first.write_bytes(b"short")
            second.write_bytes(b"longer")
            with self.assertRaises(ValueError):
                files.compare(first, second, by="unknown")


if __name__ == "__main__":
    unittest.main()

