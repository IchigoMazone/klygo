"""Behavioral contract for klygo.files.extension."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestExtension(unittest.TestCase):
    def test_final_suffix(self):
        self.assertEqual(files.extension("archive.tar.gz"), ".gz")
        self.assertEqual(files.extension("README"), "")


if __name__ == "__main__":
    unittest.main()

