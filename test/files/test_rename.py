"""Behavioral contract for klygo.files.rename."""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from klygo import files


class TestRename(unittest.TestCase):
    def test_name_and_explicit_path(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "draft.txt"
            source.touch()
            renamed = files.rename(source, "final.txt")
            self.assertEqual(renamed, root / "final.txt")
            self.assertTrue(renamed.exists())
            with self.assertRaises(ValueError):
                files.rename(renamed, renamed)


    def test_overwrite_replaces_existing_file(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "new.txt"
            target = root / "old.txt"
            source.write_text("new", encoding="utf-8")
            target.write_text("old", encoding="utf-8")
            self.assertEqual(files.rename(source, target, overwrite=True), target)
            self.assertEqual(target.read_text(encoding="utf-8"), "new")
            self.assertFalse(source.exists())

    def test_same_directory_and_failed_rename_preserve_entries(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source", root / "target"
            source.mkdir()
            target.mkdir()
            (source / "new.txt").write_text("new", encoding="utf-8")
            (target / "old.txt").write_text("old", encoding="utf-8")
            for destination in (source, source / "child", root):
                with self.subTest(destination=destination), self.assertRaises(ValueError):
                    files.rename(source, destination, overwrite=True)
            from klygo.files import filesystem

            real_replace = filesystem.os.replace

            def fail_rename(src, dst):
                if Path(src) == source:
                    raise PermissionError("locked source")
                return real_replace(src, dst)

            with patch("klygo.files.filesystem.os.replace", side_effect=fail_rename):
                with self.assertRaises(PermissionError):
                    files.rename(source, target, overwrite=True)
            self.assertEqual((source / "new.txt").read_text(encoding="utf-8"), "new")
            self.assertEqual((target / "old.txt").read_text(encoding="utf-8"), "old")

    def test_rejects_relative_symlink_relocation_that_changes_target(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "source"
            source_dir.mkdir()
            target = source_dir / "data.txt"
            target.write_text("data", encoding="utf-8")
            link = source_dir / "link.txt"
            try:
                link.symlink_to("data.txt")
            except OSError as error:
                self.skipTest(f"Symlinks unavailable: {error}")
            with self.assertRaises(ValueError):
                files.rename(link, root / "other" / "link.txt")
            self.assertEqual(link.read_text(encoding="utf-8"), "data")

    def test_no_overwrite_preserves_concurrently_created_file(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source.txt", root / "target.txt"
            source.write_text("source", encoding="utf-8")
            from klygo.files import filesystem

            operation = "rename" if os.name == "nt" else "link"
            original = getattr(filesystem.os, operation)

            def create_competing_file(src, dst):
                target.write_text("other", encoding="utf-8")
                return original(src, dst)

            with patch(f"klygo.files.filesystem.os.{operation}", side_effect=create_competing_file):
                with self.assertRaises(FileExistsError):
                    files.rename(source, target)
            self.assertEqual(source.read_text(encoding="utf-8"), "source")
            self.assertEqual(target.read_text(encoding="utf-8"), "other")


if __name__ == "__main__":
    unittest.main()

