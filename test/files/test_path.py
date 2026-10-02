"""Behavioral contract for klygo.files.path."""

import unittest
from pathlib import Path

from klygo import files


class TestPath(unittest.TestCase):
    def test_conversion_and_home_expansion(self):
        self.assertEqual(files.path("dataset"), Path("dataset"))
        self.assertEqual(files.path(Path("dataset")), Path("dataset"))
        self.assertEqual(files.path("~", expand_user=False), Path("~"))
        self.assertEqual(files.path("~"), Path.home())

    def test_invalid_arguments_raise_type_error(self):
        with self.assertRaises(TypeError):
            files.path(123)
        with self.assertRaises(TypeError):
            files.path("dataset", expand_user="yes")


if __name__ == "__main__":
    unittest.main()

