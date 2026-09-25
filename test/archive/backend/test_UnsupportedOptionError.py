"""Behavioral tests for UnsupportedOptionError."""

import unittest

from klygo.archive.backend import GZipBackend, UnsupportedOptionError


class TestUnsupportedOptionError(unittest.TestCase):
    def test_changed_unsupported_option_is_structured(self):
        with self.assertRaises(UnsupportedOptionError) as caught:
            GZipBackend().validate_option("compress", "include_root", False, True)
        error = caught.exception
        self.assertEqual(
            (error.format_name, error.operation, error.option),
            ("gz", "compress", "include_root"),
        )

    def test_unchanged_default_is_accepted(self):
        GZipBackend().validate_option("compress", "include_root", True, True)


if __name__ == "__main__":
    unittest.main()

