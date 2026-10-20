"""Behavioral contract for klygo.files.common_path."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestCommonPath(unittest.TestCase):
    def test_common_parent_and_invalid_inputs(self):
        result = files.common_path(["dataset/images/a", "dataset/labels/a"])
        self.assertEqual(result, Path("dataset"))
        with self.assertRaises(ValueError):
            files.common_path([])
        with self.assertRaises(TypeError):
            files.common_path("dataset/images")


if __name__ == "__main__":
    unittest.main()

