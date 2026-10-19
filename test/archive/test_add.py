"""Behavioral contract for klygo.archive.add."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import klygo.archive as archive
import klygo.files as files

from test.archive._support import make_archive, make_source, member_ending


class TestAdd(unittest.TestCase):
    def test_add_file_and_conflict_policy(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            packed = make_archive(root)
            extra = root / "extra.txt"
            files.save(extra, "extra", verbose=False)
            archive.add(packed, extra, verbose=False)
            self.assertIn("extra.txt", archive.list_files(packed))
            archive.add(packed, extra, on_conflict="rename", verbose=False)
            self.assertIn("extra_dup1.txt", archive.list_files(packed))

            nested_tar = root / "bundle.tar.gz"
            nested_tar.write_bytes(b"tar-gz-payload")
            archive.add(packed, nested_tar, verbose=False)
            archive.add(packed, nested_tar, on_conflict="rename", verbose=False)
            self.assertIn("bundle_dup1.tar.gz", archive.list_files(packed))

            tar_packed = make_archive(root, "packed.tar.gz", source_name="tar_source")
            archive.add(tar_packed, nested_tar, verbose=False)
            archive.add(tar_packed, nested_tar, on_conflict="rename", verbose=False)
            self.assertIn("bundle_dup1.tar.gz", archive.list_files(tar_packed))

    def test_invalid_conflict_policy_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            packed = make_archive(root)
            extra = root / "extra.txt"
            files.save(extra, "extra", verbose=False)
            with self.assertRaisesRegex(ValueError, "on_conflict"):
                archive.add(packed, extra, on_conflict="invalid", verbose=False)


if __name__ == "__main__":
    unittest.main()
