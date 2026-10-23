"""Behavioral contract for klygo.files.parent."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestParent(unittest.TestCase):
    def test_immediate_parent(self):
        self.assertEqual(files.parent("dataset/images/cat.jpg"), Path("dataset/images"))


if __name__ == "__main__":
    unittest.main()

