"""Behavioral contract for klygo.archive.merge."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import klygo.archive as archive
import klygo.files as files

from _support import make_archive, make_source, member_ending


class TestMerge(unittest.TestCase):
    def test_same_and_cross_format_merge(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = make_archive(root, "first.zip", "one")
            second = make_archive(root, "second.zip", "two")
            merged = root / "merged.zip"
            archive.merge([first, second], merged, verbose=False)
            self.assertTrue(files.is_file(merged))
            self.assertGreaterEqual(len(archive.list_files(merged)), 3)

            third_source = make_source(root, "three")
            third = root / "third.tar.gz"
            archive.compress(third_source, third, verbose=False)
            cross_format = root / "cross-format.zip"
            archive.merge([first, third], cross_format, verbose=False)
            names = archive.list_files(cross_format)
            self.assertTrue(any(name.startswith("one/") for name in names))
            self.assertTrue(any(name.startswith("three/") for name in names))


if __name__ == "__main__":
    unittest.main()
