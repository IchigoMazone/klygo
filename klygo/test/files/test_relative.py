"""Behavioral contract for klygo.files.relative."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestRelative(unittest.TestCase):
    def test_relative_path(self):
        self.assertEqual(files.relative("dataset/images/train", "dataset"), Path("images/train"))


if __name__ == "__main__":
    unittest.main()

