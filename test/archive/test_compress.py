"""Behavioral contract for klygo.archive.compress."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import klygo.archive as archive
import klygo.files as files

from _support import make_archive, make_source, member_ending


class TestCompress(unittest.TestCase):
    def test_zip_tar_and_errors(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_source(root)
            formats = (
                ("data.zip", "zip"),
                ("data.tar", "tar"),
                ("data.tar.gz", "tar.gz"),
                ("data.tar.xz", "tar.xz"),
                ("data.tar.bz2", "tar.bz2"),
            )
            for name, expected in formats:
                with self.subTest(name=name):
                    output = root / name
                    archive.compress(source, output, verbose=False)
                    self.assertTrue(files.is_file(output))
                    self.assertEqual(archive.detect_format(output), expected)
                    self.assertTrue(archive.test(output))
                    with self.assertRaises(FileExistsError):
                        archive.compress(source, output, verbose=False)

            contents_only = root / "contents-only.zip"
            archive.compress(source, contents_only, include_root=False, verbose=False)
            names = archive.list_files(contents_only)
            self.assertIn("alpha.txt", names)
            self.assertIn("nested/data.json", names)
            self.assertFalse(any(name.startswith("source/") for name in names))

            single = root / "single.txt"
            single.write_text("single-file gzip", encoding="utf-8")
            gzip_path = root / "single.txt.gz"
            archive.compress(single, gzip_path, verbose=False)
            self.assertEqual(archive.detect_format(gzip_path), "gz")
            self.assertTrue(archive.test(gzip_path))

            with self.assertRaises(FileNotFoundError):
                archive.compress(root / "missing", root / "missing.zip", verbose=False)


if __name__ == "__main__":
    unittest.main()
