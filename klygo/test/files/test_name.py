"""Behavioral contract for klygo.files.name."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestName(unittest.TestCase):
    def test_name_without_filesystem_access(self):
        self.assertEqual(files.name("dataset/images/cat.jpg"), "cat.jpg")


if __name__ == "__main__":
    unittest.main()

