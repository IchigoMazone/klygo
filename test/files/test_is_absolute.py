"""Behavioral contract for klygo.files.is_absolute."""

import unittest
from pathlib import Path

from klygo import files


class TestIsAbsolute(unittest.TestCase):
    def test_absolute_and_relative(self):
        self.assertTrue(files.is_absolute(str(Path.cwd())))
        self.assertFalse(files.is_absolute("dataset/images"))

    def test_invalid_input_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            files.is_absolute(123)


if __name__ == "__main__":
    unittest.main()

