"""Direct behavior tests for GZipBackend."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo.archive.backend import GZipBackend, UnsupportedOptionError


class TestGZipBackend(unittest.TestCase):
    def test_round_trip_member_model_and_search(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "payload.bin"
            source.write_bytes(b"payload")
            packed = root / "payload.gz"
            output = root / "out"
            backend = GZipBackend()

            backend.compress(source, packed, compresslevel=9, verbose=False)
            self.assertEqual(backend.list_files(packed), ["payload"])
            self.assertEqual(backend.search(packed, "PAY*", case_sensitive=False), ["payload"])
            backend.extract_file(packed, "payload", output)
            self.assertEqual((output / "payload").read_bytes(), b"payload")
            self.assertTrue(backend.test(packed))

    def test_directories_and_filters_are_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                GZipBackend().compress(root, root / "bad.gz", verbose=False)
            with self.assertRaises(UnsupportedOptionError):
                GZipBackend().validate_option("extract", "include", "*.txt", None)


if __name__ == "__main__":
    unittest.main()

