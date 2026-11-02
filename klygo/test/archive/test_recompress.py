"""Behavioral contract for klygo.archive.recompress."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import klygo.archive as archive
import klygo.files as files

from _support import make_archive, make_source, member_ending


class TestRecompress(unittest.TestCase):
    def test_recompress_and_overwrite_policy(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_archive(root)
            target = root / "compact.zip"
            archive.recompress(source, target, compresslevel=9, verbose=False)
            self.assertTrue(archive.test(target))
            self.assertEqual(
                set(archive.list_files(target)),
                set(archive.list_files(source)),
            )
            with self.assertRaises(FileExistsError):
                archive.recompress(source, target, verbose=False)


if __name__ == "__main__":
    unittest.main()
