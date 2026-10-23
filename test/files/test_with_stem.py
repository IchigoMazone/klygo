"""Behavioral contract for klygo.files.with_stem."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestWithStem(unittest.TestCase):
    def test_preserves_final_extension(self):
        self.assertEqual(files.with_stem("dataset/cat.jpg", "dog"), Path("dataset/dog.jpg"))


if __name__ == "__main__":
    unittest.main()

