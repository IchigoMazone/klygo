"""Behavioral contract for klygo.archive.get_info."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import klygo.archive as archive
import klygo.files as files

from _support import make_archive, make_source, member_ending


class TestGetInfo(unittest.TestCase):
    def test_metadata_contract(self):
        with TemporaryDirectory() as directory:
            packed = make_archive(Path(directory))
            result = archive.get_info(packed)
            required={"path","format","file_count","archive_size","human_archive_size","uncompressed_size"}
            self.assertTrue(required <= result.keys())
            self.assertEqual(result["format"], "zip")
            self.assertGreaterEqual(result["file_count"], 3)
            self.assertFalse(hasattr(archive, "info"))


if __name__ == "__main__":
    unittest.main()
