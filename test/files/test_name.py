"""Behavioral contract for klygo.files.name."""

import unittest
from pathlib import Path

from klygo import files


class TestName(unittest.TestCase):
    def test_name_accepts_path_and_does_not_access_filesystem(self):
        self.assertEqual(files.name("dataset/images/cat.jpg"), "cat.jpg")
        self.assertEqual(files.name(Path("missing") / "archive.tar.gz"), "archive.tar.gz")

    def test_invalid_input_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            files.name(123)


if __name__ == "__main__":
    unittest.main()

