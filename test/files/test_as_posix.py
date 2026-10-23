"""Behavioral contract for klygo.files.as_posix."""

import unittest
from pathlib import Path

from klygo import files


class TestAsPosix(unittest.TestCase):
    def test_path_representation(self):
        self.assertEqual(files.as_posix(Path("dataset") / "images" / "cat.jpg"), "dataset/images/cat.jpg")


if __name__ == "__main__":
    unittest.main()
