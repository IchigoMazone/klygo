"""Behavioral contract for klygo.archive.verify."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import klygo.archive as archive
import klygo.files as files

from _support import make_archive, make_source, member_ending


class TestVerify(unittest.TestCase):
    def test_report_contract(self):
        with TemporaryDirectory() as directory:
            packed = make_archive(Path(directory))
            result = archive.verify(packed)
            self.assertTrue(result["valid"])
            self.assertEqual(result["format"], "zip")
            self.assertGreaterEqual(result["file_count"], 3)


if __name__ == "__main__":
    unittest.main()

