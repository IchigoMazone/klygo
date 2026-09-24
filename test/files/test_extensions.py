"""Behavioral contract for klygo.files.extensions."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestExtensions(unittest.TestCase):
    def test_all_suffixes(self):
        self.assertEqual(files.extensions("archive.tar.gz"), (".tar", ".gz"))
        self.assertEqual(files.extensions("README"), ())


if __name__ == "__main__":
    unittest.main()

