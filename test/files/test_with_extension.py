"""Behavioral contract for klygo.files.with_extension."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestWithExtension(unittest.TestCase):
    def test_regular_compound_and_invalid_extensions(self):
        self.assertEqual(files.with_extension("a.jpg", "png"), Path("a.png"))
        self.assertEqual(files.with_extension("a.tar.gz", ".zip"), Path("a.zip"))
        self.assertEqual(files.with_extension("a.jpg", ""), Path("a"))
        with self.assertRaises(ValueError):
            files.with_extension("a.jpg", "bad/ext")


if __name__ == "__main__":
    unittest.main()

