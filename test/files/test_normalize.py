"""Behavioral contract for klygo.files.normalize."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestNormalize(unittest.TestCase):
    def test_dot_components(self):
        self.assertEqual(files.normalize("dataset/images/../labels"), Path("dataset/labels"))


if __name__ == "__main__":
    unittest.main()

