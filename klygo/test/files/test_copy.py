"""Behavioral contract for klygo.files.copy."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestCopy(unittest.TestCase):
    def test_file_directory_and_overwrite_policy(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            target = root / "nested" / "target.txt"
            source.write_text("one", encoding="utf-8")
            self.assertEqual(files.copy(source, target).read_text(encoding="utf-8"), "one")
            with self.assertRaises(FileExistsError):
                files.copy(source, target, overwrite=False)
            folder = root / "folder"
            folder.mkdir()
            (folder / "a.txt").touch()
            copied = files.copy(folder, root / "folder-copy")
            self.assertTrue((copied / "a.txt").is_file())


if __name__ == "__main__":
    unittest.main()

