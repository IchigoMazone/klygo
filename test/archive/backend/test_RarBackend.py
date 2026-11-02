"""Dependency and read-only contract tests for RarBackend."""

import builtins
import unittest
from unittest.mock import patch

from klygo.archive.backend import ArchiveBackend, RarBackend, UnsupportedOperationError


class TestRarBackend(unittest.TestCase):
    def test_backend_is_read_only_but_supports_password_extraction(self):
        backend = RarBackend()
        self.assertFalse(backend.capabilities.compress)
        self.assertIn("password", backend.capabilities.extract_options)
        self.assertIs(RarBackend.compress, ArchiveBackend.compress)
        with self.assertRaises(UnsupportedOperationError):
            backend.require_operation("compress")

    def test_missing_dependency_message_is_actionable(self):
        original_import = builtins.__import__

        def import_without_rarfile(name, *args, **kwargs):
            if name == "rarfile":
                raise ImportError("not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_rarfile):
            with self.assertRaisesRegex(ImportError, r"klygo\[rarfile\]"):
                RarBackend()._check_rarfile()


if __name__ == "__main__":
    unittest.main()
