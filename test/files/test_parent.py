"""Behavioral contract for klygo.files.parent."""

import unittest
from pathlib import Path

from klygo import files


class TestParent(unittest.TestCase):
    def test_immediate_parent(self):
        self.assertEqual(files.parent("dataset/images/cat.jpg"), Path("dataset/images"))
        self.assertEqual(files.parent(Path("missing") / "cat.jpg"), Path("missing"))

    def test_invalid_input_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            files.parent(123)


if __name__ == "__main__":
    unittest.main()

