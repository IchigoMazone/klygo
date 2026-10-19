"""Behavioral contract for klygo.files.copy."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

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

    def test_rejects_same_and_nested_directory_without_deleting_source(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            content = source / "keep.txt"
            content.write_text("keep", encoding="utf-8")
            for target in (source, source / "child", Path(directory)):
                with self.subTest(target=target), self.assertRaises(ValueError):
                    files.copy(source, target, overwrite=True)
                self.assertEqual(content.read_text(encoding="utf-8"), "keep")

    def test_failed_publication_restores_previous_destination(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source", root / "target"
            source.mkdir()
            target.mkdir()
            (source / "new.txt").write_text("new", encoding="utf-8")
            (target / "old.txt").write_text("old", encoding="utf-8")
            from klygo.files import filesystem

            real_replace = filesystem.os.replace

            def fail_publish(src, dst):
                if Path(src).name == "payload":
                    raise PermissionError("locked destination")
                return real_replace(src, dst)

            with patch("klygo.files.filesystem.os.replace", side_effect=fail_publish):
                with self.assertRaises(PermissionError):
                    files.copy(source, target, overwrite=True)
            self.assertEqual((source / "new.txt").read_text(encoding="utf-8"), "new")
            self.assertEqual((target / "old.txt").read_text(encoding="utf-8"), "old")

    def test_failed_rollback_retains_previous_destination_as_backup(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source", root / "target"
            source.mkdir()
            target.mkdir()
            (source / "new.txt").write_text("new", encoding="utf-8")
            (target / "old.txt").write_text("old", encoding="utf-8")
            from klygo.files import filesystem

            real_replace = filesystem.os.replace

            def fail_publish_and_rollback(src, dst):
                if Path(src).name in ("payload", "previous"):
                    raise PermissionError("locked")
                return real_replace(src, dst)

            with patch("klygo.files.filesystem.os.replace", side_effect=fail_publish_and_rollback):
                with self.assertRaisesRegex(OSError, "backup retained"):
                    files.copy(source, target, overwrite=True)
            backups = list(root.glob(".target.*/previous/old.txt"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(encoding="utf-8"), "old")
            self.assertEqual((source / "new.txt").read_text(encoding="utf-8"), "new")

    def test_no_overwrite_rejects_concurrent_file_creation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source.txt", root / "target.txt"
            source.write_text("source", encoding="utf-8")
            from klygo.files import filesystem

            real_link = filesystem.os.link

            def create_competing_file(src, dst):
                Path(dst).write_text("other", encoding="utf-8")
                return real_link(src, dst)

            with patch("klygo.files.filesystem.os.link", side_effect=create_competing_file):
                with self.assertRaises(FileExistsError):
                    files.copy(source, target)
            self.assertEqual(source.read_text(encoding="utf-8"), "source")
            self.assertEqual(target.read_text(encoding="utf-8"), "other")

    def test_copy_relative_symlink_preserves_its_referent(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "source"
            source_dir.mkdir()
            original = source_dir / "data.txt"
            original.write_text("data", encoding="utf-8")
            link = source_dir / "link.txt"
            try:
                link.symlink_to("data.txt")
            except OSError as error:
                self.skipTest(f"Symlinks unavailable: {error}")
            copied = files.copy(link, root / "other" / "link.txt")
            self.assertTrue(copied.is_symlink())
            self.assertEqual(copied.read_text(encoding="utf-8"), "data")


if __name__ == "__main__":
    unittest.main()

