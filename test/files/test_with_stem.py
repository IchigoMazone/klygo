"""Behavioral contract for klygo.files.with_stem."""

import unittest
from pathlib import Path

from klygo import files


class TestWithStem(unittest.TestCase):
    def test_preserves_final_extension_without_filesystem_access(self):
        self.assertEqual(files.with_stem("dataset/cat.jpg", "dog"), Path("dataset/dog.jpg"))
        self.assertEqual(files.with_stem(Path("missing") / "archive.tar.gz", "backup"), Path("missing/backup.gz"))

    def test_new_stem_must_be_string(self):
        with self.assertRaises(TypeError):
            files.with_stem("cat.jpg", 123)


if __name__ == "__main__":
    unittest.main()

