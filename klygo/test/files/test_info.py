"""Behavioral contract for klygo.files.info."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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


if __name__ == "__main__":
    unittest.main()

