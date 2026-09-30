"""Behavioral contract for klygo.files.compound_extension."""

import unittest
from pathlib import Path

from klygo import files


class TestCompoundExtension(unittest.TestCase):
    def test_joined_suffixes(self):
        self.assertEqual(files.compound_extension("archive.tar.gz"), ".tar.gz")
        self.assertEqual(files.compound_extension("backup.sql.tar.xz"), ".sql.tar.xz")
        self.assertEqual(files.compound_extension("photo.jpg"), ".jpg")
        self.assertEqual(files.compound_extension("README"), "")

    def test_path_input_and_nonexistent_path_do_not_access_filesystem(self):
        self.assertEqual(
            files.compound_extension(Path("missing") / "release.min.js.map"),
            ".min.js.map",
        )

    def test_hidden_files_and_trailing_dots_have_no_extension(self):
        self.assertEqual(files.compound_extension(".gitignore"), "")
        self.assertEqual(files.compound_extension("filename."), "")

    def test_invalid_input_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            files.compound_extension(123)


if __name__ == "__main__":
    unittest.main()

