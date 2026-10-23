"""Behavioral contract for klygo.files.unique_path."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from unittest.mock import patch

from klygo import files


class TestUniquePath(unittest.TestCase):
    def test_available_numbered_and_compound_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            available = root / "free.txt"
            self.assertEqual(files.unique_path(available), available)
            existing = root / "archive.tar.gz"
            existing.touch()
            self.assertEqual(files.unique_path(existing), root / "archive_1.tar.gz")
            with self.assertRaises(ValueError):
                files.unique_path(existing, start=-1)

    def test_broken_links_occupy_both_original_and_numbered_names(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "archive.tar.gz"
            numbered = root / "archive_1.tar.gz"
            try:
                source.symlink_to(root / "missing")
                numbered.symlink_to(root / "missing")
            except OSError as error:
                if getattr(error, "winerror", None) == 1314 or error.errno in (1, 13, 95):
                    self.skipTest(f"Symlinks unavailable: {error}")
                raise
            self.assertEqual(files.unique_path(source), root / "archive_2.tar.gz")
            self.assertFalse((root / "archive_2.tar.gz").exists())

    def test_unavailable_targets_do_not_make_link_names_available(self):
        occupied = {Path("archive.tar.gz"), Path("archive_1.tar.gz")}
        with patch.object(Path, "exists", return_value=False), \
                patch.object(Path, "is_symlink", autospec=True, side_effect=lambda path: path in occupied):
            self.assertEqual(files.unique_path("archive.tar.gz"), Path("archive_2.tar.gz"))


if __name__ == "__main__":
    unittest.main()

