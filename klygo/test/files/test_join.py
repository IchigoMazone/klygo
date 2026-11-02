"""Behavioral contract for klygo.files.join."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestJoin(unittest.TestCase):
    def test_join_and_missing_parts(self):
        self.assertEqual(files.join("dataset", "images", "cat.jpg"), Path("dataset/images/cat.jpg"))
        with self.assertRaises(ValueError):
            files.join()


if __name__ == "__main__":
    unittest.main()

