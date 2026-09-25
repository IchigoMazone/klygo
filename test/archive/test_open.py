"""Behavioral contract for klygo.archive.open."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import klygo.archive as archive
import klygo.files as files

from _support import make_archive, make_source, member_ending


class TestOpen(unittest.TestCase):
    def test_context_manager(self):
        with TemporaryDirectory() as directory:
            packed = make_archive(Path(directory))
            with archive.open(packed) as opened:
                self.assertIsInstance(opened, archive.ArchiveFile)
                self.assertEqual(opened.format, "zip")


if __name__ == "__main__":
    unittest.main()

