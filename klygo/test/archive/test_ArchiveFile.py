"""Behavioral contract for klygo.archive.ArchiveFile."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import klygo.archive as archive
import klygo.files as files

from _support import make_archive, make_source, member_ending


class TestArchiveFile(unittest.TestCase):
    def test_object_interface(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            packed = make_archive(root)
            opened = archive.ArchiveFile(packed)
            self.assertEqual(opened.list_files(), archive.list_files(packed))
            self.assertEqual(opened.search("*.txt"), archive.search(packed, "*.txt"))
            self.assertTrue(opened.test())
            output = root / "output"
            opened.extract(output, verbose=False)
            self.assertTrue(files.find(output))


if __name__ == "__main__":
    unittest.main()

