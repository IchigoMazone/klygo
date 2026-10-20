"""Behavioral contract for klygo.archive.extract_file."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import klygo.archive as archive
import klygo.files as files

from _support import make_archive, make_source, member_ending


class TestExtractFile(unittest.TestCase):
    def test_single_member_and_missing_member(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            packed = make_archive(root)
            member = member_ending(archive.list_files(packed), "data.json")
            output = root / "single"
            archive.extract_file(packed, member, output)
            self.assertTrue(files.is_file(output / "data.json"))
            with self.assertRaises(FileExistsError):
                archive.extract_file(packed, member, output)
            with self.assertRaises(KeyError):
                archive.extract_file(packed, "missing.txt", output)


if __name__ == "__main__":
    unittest.main()

