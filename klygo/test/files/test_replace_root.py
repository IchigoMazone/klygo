"""Behavioral contract for klygo.files.replace_root."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestReplaceRoot(unittest.TestCase):
    def test_mapping_and_outside_error(self):
        result = files.replace_root("dataset/images/train/a.jpg", "dataset/images", "dataset/labels")
        self.assertEqual(result, Path("dataset/labels/train/a.jpg"))
        with self.assertRaises(ValueError):
            files.replace_root("outside/a.jpg", "dataset/images", "dataset/labels")


if __name__ == "__main__":
    unittest.main()

