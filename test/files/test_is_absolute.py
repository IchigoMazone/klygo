"""Behavioral contract for klygo.files.is_absolute."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestIsAbsolute(unittest.TestCase):
    def test_absolute_and_relative(self):
        self.assertTrue(files.is_absolute(Path.cwd()))
        self.assertFalse(files.is_absolute("dataset/images"))


if __name__ == "__main__":
    unittest.main()

