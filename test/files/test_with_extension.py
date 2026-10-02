"""Behavioral contract for klygo.files.with_extension."""

import unittest
from pathlib import Path

from klygo import files


class TestWithExtension(unittest.TestCase):
    def test_regular_compound_and_missing_extensions(self):
        self.assertEqual(files.with_extension("a.jpg", "png"), Path("a.png"))
        self.assertEqual(files.with_extension("a.tar.gz", ".zip"), Path("a.zip"))
        self.assertEqual(files.with_extension("a.jpg", ""), Path("a"))
        self.assertEqual(files.with_extension(Path("missing") / "README", "md"), Path("missing/README.md"))

    def test_rejects_invalid_extensions_and_types(self):
        with self.assertRaises(ValueError):
            files.with_extension("a.jpg", "bad/ext")
        with self.assertRaises(TypeError):
            files.with_extension("a.jpg", 123)


if __name__ == "__main__":
    unittest.main()

