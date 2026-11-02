"""Behavioral tests for BackendCapabilities."""

import unittest
from dataclasses import FrozenInstanceError

from klygo.archive.backend import BackendCapabilities, ZipBackend


class TestBackendCapabilities(unittest.TestCase):
    def test_defaults_are_read_only(self):
        capabilities = BackendCapabilities()
        self.assertFalse(capabilities.compress)
        self.assertEqual(capabilities.compress_options, frozenset())
        with self.assertRaises(FrozenInstanceError):
            capabilities.compress = True

    def test_zip_declares_real_options(self):
        capabilities = ZipBackend.capabilities
        self.assertTrue(all((capabilities.compress, capabilities.add, capabilities.remove)))
        self.assertIn("method", capabilities.compress_options)
        self.assertIn("password", capabilities.extract_options)
        self.assertEqual(capabilities.add_options, frozenset({"on_conflict"}))


if __name__ == "__main__":
    unittest.main()

