"""Behavioral contract for klygo.files.is_symlink."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestIsSymlink(unittest.TestCase):
    def test_regular_and_missing_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            regular = root / "regular.txt"
            regular.write_text("data", encoding="utf-8")
            self.assertFalse(files.is_symlink(regular))
            self.assertFalse(files.is_symlink(root / "missing"))

    def test_broken_symlink_when_supported(self):
        with TemporaryDirectory() as directory:
            link = Path(directory) / "link"
            try:
                link.symlink_to("missing")
            except (OSError, NotImplementedError):
                self.skipTest("Creating symlinks is unavailable on this host")
            self.assertTrue(files.is_symlink(link))


if __name__ == "__main__":
    unittest.main()
