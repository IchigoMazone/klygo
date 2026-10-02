"""Behavioral contract for klygo.files.stem."""

import unittest
from pathlib import Path

from klygo import files


class TestStem(unittest.TestCase):
    def test_only_final_suffix_is_removed_without_filesystem_access(self):
        self.assertEqual(files.stem("archive.tar.gz"), "archive.tar")
        self.assertEqual(files.stem(Path("missing") / "photo.jpg"), "photo")
        self.assertEqual(files.stem("README"), "README")

    def test_invalid_input_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            files.stem(123)


if __name__ == "__main__":
    unittest.main()

