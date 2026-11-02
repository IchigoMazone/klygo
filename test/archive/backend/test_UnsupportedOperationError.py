"""Behavioral tests for UnsupportedOperationError."""

import unittest

from klygo.archive.backend import GZipBackend, UnsupportedOperationError


class TestUnsupportedOperationError(unittest.TestCase):
    def test_structured_attributes_and_standard_base(self):
        with self.assertRaises(UnsupportedOperationError) as caught:
            GZipBackend().require_operation("add")
        error = caught.exception
        self.assertIsInstance(error, NotImplementedError)
        self.assertEqual((error.format_name, error.operation), ("gz", "add"))
        self.assertIn("does not support 'add'", str(error))


if __name__ == "__main__":
    unittest.main()

