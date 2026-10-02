"""Behavioral contract for klygo.files.normalize."""

import unittest
from pathlib import Path

from klygo import files


class TestNormalize(unittest.TestCase):
    def test_dot_components_and_path_input(self):
        self.assertEqual(files.normalize("dataset/images/../labels"), Path("dataset/labels"))
        self.assertEqual(files.normalize(Path("dataset") / "." / "labels"), Path("dataset/labels"))

    def test_invalid_input_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            files.normalize(123)


if __name__ == "__main__":
    unittest.main()

