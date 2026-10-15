"""Behavioral contract for klygo.files.info."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from unittest.mock import patch

from klygo import files


class TestInfo(unittest.TestCase):
    def test_file_metadata(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "sample.txt"
            target.write_text("abc", encoding="utf-8")
            result = files.info(target)
            self.assertEqual(result["name"], "sample.txt")
            self.assertEqual(result["size"], 3)
            self.assertTrue(result["is_file"])
            self.assertEqual(len(result["hash"]), 32)

    def test_optional_hash_skips_content_reads_and_preserves_keys(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "data.bin"
            source.write_bytes(b"content")
            regular = files.info(source)
            with patch("klygo.files.metadata.hash", side_effect=AssertionError("unexpected hash")):
                metadata = files.info(source, include_hash=False)
            self.assertIsNone(metadata["hash"])
            self.assertEqual(metadata.keys(), regular.keys())
            self.assertEqual(metadata["size"], 7)
            with self.assertRaises(TypeError):
                files.info(source, include_hash=0)


if __name__ == "__main__":
    unittest.main()

