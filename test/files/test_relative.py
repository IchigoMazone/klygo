"""Behavioral contract for klygo.files.relative."""

import unittest
from pathlib import Path

from klygo import files


class TestRelative(unittest.TestCase):
    def test_relative_path_and_parent_traversal(self):
        self.assertEqual(files.relative("dataset/images/train", "dataset"), Path("images/train"))
        self.assertEqual(files.relative(Path("dataset/labels"), "dataset/images"), Path("../labels"))

    def test_invalid_input_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            files.relative(123)


if __name__ == "__main__":
    unittest.main()

