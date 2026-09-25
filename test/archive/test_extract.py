"""Behavioral contract for klygo.archive.extract."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import klygo.archive as archive
import klygo.files as files

from _support import make_archive, make_source, member_ending


class TestExtract(unittest.TestCase):
    def test_filters_overwrite_and_zip_slip(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            packed = make_archive(root)
            output = root / "output"
            archive.extract(packed, output, include="*.txt", verbose=False)
            self.assertTrue(any(p.name == "alpha.txt" for p in files.find(output)))
            self.assertFalse(any(p.suffix == ".json" for p in files.find(output)))
            with self.assertRaises(FileExistsError):
                archive.extract(packed, output, include="*.txt", verbose=False)
            malicious = root / "unsafe.zip"
            with ZipFile(malicious, "w") as stream:
                stream.writestr("../escape.txt", "unsafe")
            with self.assertRaises(ValueError):
                archive.extract(malicious, root / "safe", verbose=False)


if __name__ == "__main__":
    unittest.main()

