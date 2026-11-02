"""Behavioral contract for klygo.files.remove."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from unittest.mock import patch

from klygo import files


class TestRemove(unittest.TestCase):
    def test_file_directory_and_missing_policy(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            item = root / "item.txt"
            item.touch()
            files.remove(item)
            self.assertFalse(item.exists())
            folder = root / "folder"
            folder.mkdir()
            (folder / "child").touch()
            files.remove(folder)
            self.assertFalse(folder.exists())
            with self.assertRaises(FileNotFoundError):
                files.remove(root / "missing", missing_ok=False)

    def test_live_and_broken_links_leave_targets_untouched(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.txt"
            target.write_text("keep", encoding="utf-8")
            link = root / "link.txt"
            try:
                link.symlink_to(target)
            except OSError as error:
                if getattr(error, "winerror", None) == 1314 or error.errno in (1, 13, 95):
                    self.skipTest(f"Symlinks unavailable: {error}")
                raise
            files.remove(link, missing_ok=False)
            self.assertEqual(target.read_text(encoding="utf-8"), "keep")
            for missing_ok in (True, False):
                link.symlink_to(root / "missing")
                files.remove(link, missing_ok=missing_ok)
                self.assertFalse(link.is_symlink())
            folder = root / "folder"
            folder.mkdir()
            child = folder / "child.txt"
            child.touch()
            link.symlink_to(folder, target_is_directory=True)
            files.remove(link)
            self.assertTrue(child.exists())

    def test_broken_link_is_removed_when_target_is_absent(self):
        link = Path("broken-link")
        with patch.object(Path, "exists", return_value=False), \
                patch.object(Path, "is_symlink", return_value=True), \
                patch.object(Path, "is_file", return_value=False), \
                patch.object(Path, "unlink", autospec=True) as unlink, \
                patch("klygo.files.filesystem.shutil.rmtree") as remove_tree:
            files.remove(link, missing_ok=False)
        unlink.assert_called_once_with(link)
        remove_tree.assert_not_called()


if __name__ == "__main__":
    unittest.main()

