"""Behavioral contract for klygo.files.iter_find."""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from klygo import files


class TestIterFind(unittest.TestCase):
    def test_filters_prunes_exclusions_and_controls_recursion(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("nested", "cache-data", "folder.jpg"):
                (root / name).mkdir()
            for name in ("top.jpg", "top.txt", "nested/child.jpg", "cache-data/hidden.jpg"):
                (root / name).touch()
            with patch("os.scandir", wraps=os.scandir) as scan:
                result = set(files.iter_find(root, "*.jpg", exclude_dirs=("cache*",)))
            self.assertEqual(result, {root / "top.jpg", root / "nested/child.jpg"})
            self.assertNotIn(root / "cache-data", [Path(call.args[0]) for call in scan.call_args_list])
            self.assertEqual(list(files.iter_find(root, "*.jpg", recursive=False)), [root / "top.jpg"])

    def test_traversal_is_lazy_and_errors_propagate(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            result = files.iter_find(root)
            late = root / "late.txt"
            late.touch()
            self.assertEqual(list(result), [late])

            def fail_scan(*args, **kwargs):
                kwargs["onerror"](PermissionError("unreadable directory"))

            with patch("klygo.files.filesystem.os.walk", side_effect=fail_scan) as walk:
                result = files.iter_find(root)
                walk.assert_not_called()
                with self.assertRaises(PermissionError):
                    next(result)

    def test_invalid_options_and_roots(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for pattern in ("", "nested/*.jpg", "nested\\*.jpg"):
                with self.subTest(pattern=pattern), self.assertRaises(ValueError):
                    files.iter_find(root, pattern)
            with self.assertRaises(TypeError):
                files.iter_find(root, exclude_dirs="cache")
            with self.assertRaises(TypeError):
                files.iter_find(root, exclude_dirs=(None,))
            with self.assertRaises(ValueError):
                files.iter_find(root, exclude_dirs=("a/b",))
            with self.assertRaises(TypeError):
                files.iter_find(root, recursive=1)
            with self.assertRaises(FileNotFoundError):
                files.iter_find(root / "missing")
            source = root / "file.txt"
            source.touch()
            with self.assertRaises(ValueError):
                files.iter_find(source)

    def test_directory_links_are_not_traversed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            actual = root / "actual"
            actual.mkdir()
            (actual / "item.txt").touch()
            try:
                (root / "alias").symlink_to(actual, target_is_directory=True)
                (root / "file-link.txt").symlink_to(actual / "item.txt")
            except OSError as error:
                if getattr(error, "winerror", None) == 1314 or error.errno in (1, 13, 95):
                    self.skipTest(f"Symlinks unavailable: {error}")
                raise
            self.assertEqual(set(files.iter_find(root, "*.txt")), {actual / "item.txt", root / "file-link.txt"})


if __name__ == "__main__":
    unittest.main()
