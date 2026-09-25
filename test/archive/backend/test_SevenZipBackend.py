"""Dependency and capability tests for SevenZipBackend."""

import builtins
import unittest
from unittest.mock import patch

from klygo.archive.backend import ArchiveBackend, SevenZipBackend


class TestSevenZipBackend(unittest.TestCase):
    def test_capabilities_match_implemented_operations(self):
        backend = SevenZipBackend()
        self.assertTrue(backend.capabilities.compress)
        self.assertIn("password", backend.capabilities.extract_options)
        self.assertIs(SevenZipBackend.add, ArchiveBackend.add)

    def test_missing_dependency_message_is_actionable(self):
        original_import = builtins.__import__

        def import_without_py7zr(name, *args, **kwargs):
            if name == "py7zr":
                raise ImportError("not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_py7zr):
            with self.assertRaisesRegex(ImportError, r"klygo\[py7zr\]"):
                SevenZipBackend()._check_py7zr()


if __name__ == "__main__":
    unittest.main()
