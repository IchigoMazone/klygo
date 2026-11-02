"""Behavioral contract for klygo.files.info."""

import io
import unittest
from contextlib import redirect_stdout
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
            self.assertNotIn("hash", result)

    def test_info_skips_content_reads_and_verbose_prints_table(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "data.bin"
            source.write_bytes(b"content")
            with patch("klygo.files.metadata.hash", side_effect=AssertionError("unexpected hash")):
                quiet_output = io.StringIO()
                with redirect_stdout(quiet_output):
                    metadata = files.info(source)
                output = io.StringIO()
                with redirect_stdout(output):
                    displayed = files.info(source, verbose=True)
            self.assertEqual(metadata, displayed)
            self.assertEqual(quiet_output.getvalue(), "")
            self.assertEqual(metadata["size"], 7)
            self.assertIn("File information", output.getvalue())
            self.assertIn("data.bin", output.getvalue())
            self.assertIn("size", output.getvalue())
            with self.assertRaises(TypeError):
                files.info(source, verbose=0)

    def test_directory_metadata(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data.bin").write_bytes(b"content")
            metadata = files.info(root)
            self.assertTrue(metadata["is_dir"])
            self.assertEqual(metadata["size"], 7)
            self.assertNotIn("hash", metadata)


if __name__ == "__main__":
    unittest.main()

