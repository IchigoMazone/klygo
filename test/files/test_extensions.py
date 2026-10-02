"""Behavioral contract for klygo.files.extensions."""

import unittest
from pathlib import Path

from klygo import files


class TestExtensions(unittest.TestCase):
    def test_all_suffixes(self):
        self.assertEqual(files.extensions("archive.tar.gz"), (".tar", ".gz"))
        self.assertEqual(files.extensions("README"), ())

    def test_path_input_and_edge_case_names(self):
        self.assertEqual(
            files.extensions(Path("missing") / "release.min.js.map"),
            (".min", ".js", ".map"),
        )
        self.assertEqual(files.extensions(".gitignore"), ())
        self.assertEqual(files.extensions("filename."), ())

    def test_invalid_input_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            files.extensions(123)


if __name__ == "__main__":
    unittest.main()

