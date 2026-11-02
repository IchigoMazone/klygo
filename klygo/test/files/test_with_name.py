"""Behavioral contract for klygo.files.with_name."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestWithName(unittest.TestCase):
    def test_replaces_name_only(self):
        self.assertEqual(files.with_name("dataset/cat.jpg", "dog.png"), Path("dataset/dog.png"))


if __name__ == "__main__":
    unittest.main()

