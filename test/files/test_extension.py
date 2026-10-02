"""Behavioral contract for klygo.files.extension."""

import unittest
from pathlib import Path

from klygo import files


class TestExtension(unittest.TestCase):
    def test_final_suffix(self):
        self.assertEqual(files.extension("archive.tar.gz"), ".gz")
        self.assertEqual(files.extension("README"), "")

    def test_path_input_and_edge_case_names(self):
        self.assertEqual(files.extension(Path("missing") / "release.min.js.map"), ".map")
        self.assertEqual(files.extension(".gitignore"), "")
        self.assertEqual(files.extension("filename."), "")

    def test_invalid_input_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            files.extension(123)


if __name__ == "__main__":
    unittest.main()
