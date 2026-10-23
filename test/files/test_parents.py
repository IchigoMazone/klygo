"""Behavioral contract for klygo.files.parents."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestParents(unittest.TestCase):
    def test_order_and_type(self):
        result = files.parents("dataset/images/train/a.jpg")
        self.assertIsInstance(result, tuple)
        self.assertEqual(result[0], Path("dataset/images/train"))


if __name__ == "__main__":
    unittest.main()

