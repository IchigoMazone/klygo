"""Behavioral contract for klygo.files.path."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestPath(unittest.TestCase):
    def test_conversion_and_home_expansion(self):
        self.assertEqual(files.path("dataset"), Path("dataset"))
        self.assertEqual(files.path("~", expand_user=False), Path("~"))
        self.assertEqual(files.path("~"), Path.home())


if __name__ == "__main__":
    unittest.main()

