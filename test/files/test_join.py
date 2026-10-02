"""Behavioral contract for klygo.files.join."""

import unittest
from pathlib import Path

from klygo import files


class TestJoin(unittest.TestCase):
    def test_joins_string_and_path_parts(self):
        self.assertEqual(
            files.join("dataset", Path("images"), "cat.jpg"),
            Path("dataset/images/cat.jpg"),
        )

    def test_rejects_missing_or_invalid_parts(self):
        with self.assertRaises(ValueError):
            files.join()
        with self.assertRaises(TypeError):
            files.join("dataset", 123)


if __name__ == "__main__":
    unittest.main()

