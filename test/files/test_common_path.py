"""Behavioral contract for klygo.files.common_path."""

import unittest
from pathlib import Path

from klygo import files


class TestCommonPath(unittest.TestCase):
    def test_common_parent_and_invalid_inputs(self):
        result = files.common_path(["dataset/images/a", "dataset/labels/a"])
        self.assertEqual(result, Path("dataset"))
        with self.assertRaises(ValueError):
            files.common_path([])
        with self.assertRaises(TypeError):
            files.common_path("dataset/images")

    def test_path_iterable_and_invalid_items(self):
        paths = (Path("dataset") / split / "image.jpg" for split in ("train", "val"))
        self.assertEqual(files.common_path(paths), Path("dataset"))
        self.assertEqual(files.common_path([Path("dataset/train/image.jpg")]), Path("dataset/train/image.jpg"))
        with self.assertRaises(TypeError):
            files.common_path(["dataset/train", 1])


if __name__ == "__main__":
    unittest.main()

