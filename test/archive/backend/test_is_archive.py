"""Behavioral tests for backend-level is_archive."""

import unittest

from klygo.archive.backend import is_archive


class TestIsArchive(unittest.TestCase):
    def test_probe_never_raises_for_common_inputs(self):
        self.assertTrue(is_archive("data.zip"))
        self.assertTrue(is_archive("data.tar.xz"))
        self.assertFalse(is_archive("README.md"))
        self.assertFalse(is_archive(""))


if __name__ == "__main__":
    unittest.main()

