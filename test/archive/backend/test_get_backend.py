"""Behavioral tests for get_backend and its registry aliases."""

import unittest

from klygo.archive.backend import (
    GZipBackend,
    RarBackend,
    SevenZipBackend,
    TarBackend,
    ZipBackend,
    get_backend,
)


class TestGetBackend(unittest.TestCase):
    def test_registered_formats_construct_expected_types(self):
        cases = {
            "zip": ZipBackend,
            "tar": TarBackend,
            "gz": GZipBackend,
            "7z": SevenZipBackend,
            "rar": RarBackend,
        }
        for format_name, expected in cases.items():
            with self.subTest(format=format_name):
                self.assertIsInstance(get_backend("unused", format_name), expected)

    def test_alias_is_canonical_and_instances_are_not_shared(self):
        first = get_backend("unused", "tgz")
        second = get_backend("unused", "tgz")
        self.assertEqual(first.format_name, "tar.gz")
        self.assertIsNot(first, second)

    def test_unknown_format_raises(self):
        with self.assertRaisesRegex(ValueError, "No backend available"):
            get_backend("unused", "unknown")


if __name__ == "__main__":
    unittest.main()

