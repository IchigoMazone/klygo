"""Behavioral contract for klygo.files.is_within."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestIsWithin(unittest.TestCase):
    def test_inside_outside_and_dot_components(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertTrue(files.is_within(root / "images" / "a.jpg", root))
            self.assertFalse(files.is_within(root / ".." / "outside", root))


if __name__ == "__main__":
    unittest.main()

