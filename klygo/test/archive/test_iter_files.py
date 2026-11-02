"""Behavioral contract for klygo.archive.iter_files."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import klygo.archive as archive
import klygo.files as files

from _support import make_archive, make_source, member_ending


class TestIterFiles(unittest.TestCase):
    def test_lazy_iteration_matches_list(self):
        with TemporaryDirectory() as directory:
            packed = make_archive(Path(directory))
            iterator = archive.iter_files(packed)
            self.assertTrue(hasattr(iterator, "__next__"))
            self.assertEqual(list(iterator), archive.list_files(packed))


if __name__ == "__main__":
    unittest.main()

