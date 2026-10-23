"""Behavioral contract for klygo.files.hash."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files


class TestHash(unittest.TestCase):
    def test_algorithms_and_invalid_target(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "data.bin"
            target.write_bytes(b"abc")
            self.assertEqual(files.hash(target), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
            self.assertEqual(files.hash(target, "md5"), "900150983cd24fb0d6963f7d28e17f72")
            with self.assertRaises(ValueError):
                files.hash(directory)


if __name__ == "__main__":
    unittest.main()

