"""Behavioral contract for klygo.files.parents."""

import unittest
from pathlib import Path

from klygo import files


class TestParents(unittest.TestCase):
    def test_order_type_and_path_input(self):
        result = files.parents("dataset/images/train/a.jpg")
        self.assertIsInstance(result, tuple)
        self.assertEqual(result[:3], (Path("dataset/images/train"), Path("dataset/images"), Path("dataset")))
        self.assertEqual(files.parents(Path("missing") / "file.txt")[0], Path("missing"))

    def test_invalid_input_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            files.parents(123)


if __name__ == "__main__":
    unittest.main()

