"""Behavioral contract for klygo.files.is_within."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestIsWithin(unittest.TestCase):
    def test_inside_outside_root_and_dot_components(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertTrue(files.is_within(str(root / "images" / "a.jpg"), root))
            self.assertTrue(files.is_within(root, root))
            self.assertFalse(files.is_within(root / ".." / "outside", root))

    def test_resolve_paths_must_be_boolean(self):
        with self.assertRaises(TypeError):
            files.is_within("child", "root", resolve_paths="yes")


if __name__ == "__main__":
    unittest.main()

