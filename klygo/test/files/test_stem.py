"""Behavioral contract for klygo.files.stem."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestStem(unittest.TestCase):
    def test_only_final_suffix_is_removed(self):
        self.assertEqual(files.stem("archive.tar.gz"), "archive.tar")


if __name__ == "__main__":
    unittest.main()

