"""Direct behavior tests for ZipBackend."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.files as files
from klygo.archive.backend import ZipBackend


class TestZipBackend(unittest.TestCase):
    def test_round_trip_search_and_metadata(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Report.TXT"
            files.save(source, "zip content", verbose=False)
            packed = root / "report.zip"
            output = root / "output"

            backend = ZipBackend()
            backend.compress(source, packed, method="deflated", verbose=False)
            self.assertEqual(backend.list_files(packed), ["Report.TXT"])
            self.assertEqual(backend.search(packed, "*.txt", case_sensitive=False), ["Report.TXT"])
            self.assertEqual(backend.get_info(packed)["format"], "zip")
            self.assertTrue(backend.test(packed))

            backend.extract_file(packed, "Report.TXT", output)
            self.assertEqual(files.load(output / "Report.TXT", verbose=False), "zip content")

    def test_rename_conflicts_are_unique(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "item.txt"
            files.save(source, "value", verbose=False)
            packed = root / "items.zip"
            backend = ZipBackend()
            backend.compress(source, packed, verbose=False)
            backend.add(packed, [source], on_conflict="rename", verbose=False)
            backend.add(packed, [source], on_conflict="rename", verbose=False)
            self.assertEqual(
                backend.list_files(packed),
                ["item.txt", "item_dup1.txt", "item_dup2.txt"],
            )


if __name__ == "__main__":
    unittest.main()

