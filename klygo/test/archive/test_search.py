"""Behavioral contract for klygo.archive.search."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import klygo.archive as archive
import klygo.files as files

from _support import make_archive, make_source, member_ending


class TestSearch(unittest.TestCase):
    def test_glob_regex_and_case_modes(self):
        with TemporaryDirectory() as directory:
            packed = make_archive(Path(directory))
            self.assertEqual(len(archive.search(packed, "*.txt")), 1)
            self.assertEqual(len(archive.search(packed, r".*data\.json$", regex=True)), 1)
            self.assertEqual(len(archive.search(packed, "*.TXT", case_sensitive=False)), 1)


if __name__ == "__main__":
    unittest.main()
