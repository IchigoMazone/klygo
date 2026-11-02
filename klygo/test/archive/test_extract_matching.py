"""Behavioral contract for klygo.archive.extract_matching."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import klygo.archive as archive
import klygo.files as files

from _support import make_archive, make_source, member_ending


class TestExtractMatching(unittest.TestCase):
    def test_wildcard_extraction(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            packed = make_archive(root)
            output = root / "logs"
            archive.extract_matching(packed, "*.log", output)
            found = files.find(output)
            self.assertEqual([p.name for p in found], ["beta.log"])


if __name__ == "__main__":
    unittest.main()

