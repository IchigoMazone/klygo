"""Behavioral contract for klygo.archive.list_files."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import klygo.archive as archive
import klygo.files as files

from _support import make_archive, make_source, member_ending


class TestListFiles(unittest.TestCase):
    def test_lists_all_members(self):
        with TemporaryDirectory() as directory:
            packed = make_archive(Path(directory))
            names = archive.list_files(packed)
            self.assertTrue(any(n.endswith("alpha.txt") for n in names))
            self.assertTrue(any(n.endswith("data.json") for n in names))
            self.assertFalse(hasattr(archive, "list"))


if __name__ == "__main__":
    unittest.main()
