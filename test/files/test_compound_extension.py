"""Behavioral contract for klygo.files.compound_extension."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestCompoundExtension(unittest.TestCase):
    def test_joined_suffixes(self):
        self.assertEqual(files.compound_extension("archive.tar.gz"), ".tar.gz")
        self.assertEqual(files.compound_extension("README"), "")


if __name__ == "__main__":
    unittest.main()

