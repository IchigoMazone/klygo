"""Behavioral contract for klygo.archive.remove."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import klygo.archive as archive
import klygo.files as files

from _support import make_archive, make_source, member_ending


class TestRemove(unittest.TestCase):
    def test_remove_member_and_missing_member(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            packed = make_archive(root)
            member = member_ending(archive.list_files(packed), "alpha.txt")
            archive.remove(packed, member)
            self.assertNotIn(member, archive.list_files(packed))
            with self.assertRaises(KeyError):
                archive.remove(packed, "missing.txt")


if __name__ == "__main__":
    unittest.main()

