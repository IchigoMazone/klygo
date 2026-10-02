"""Behavioral contract for klygo.files.with_name."""

import unittest
from pathlib import Path

from klygo import files


class TestWithName(unittest.TestCase):
    def test_replaces_name_only_without_filesystem_access(self):
        self.assertEqual(files.with_name("dataset/cat.jpg", "dog.png"), Path("dataset/dog.png"))
        self.assertEqual(files.with_name(Path("missing") / "cat.jpg", "dog.jpg"), Path("missing/dog.jpg"))

    def test_new_name_must_be_string(self):
        with self.assertRaises(TypeError):
            files.with_name("cat.jpg", 123)


if __name__ == "__main__":
    unittest.main()

