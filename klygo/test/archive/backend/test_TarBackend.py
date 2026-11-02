"""Direct behavior tests for TarBackend."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.files as files
from klygo.archive.backend import TarBackend, UnsupportedOptionError


class TestTarBackend(unittest.TestCase):
    def test_capabilities_depend_on_selected_variant(self):
        self.assertNotIn("compresslevel", TarBackend("tar").capabilities.compress_options)
        self.assertIn("compresslevel", TarBackend("tar.gz").capabilities.compress_options)

    def test_raw_tar_rejects_changed_compression_level(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "item.txt"
            files.save(source, "value", verbose=False)
            with self.assertRaises(UnsupportedOptionError):
                TarBackend("tar").compress(
                    source, root / "item.tar", compresslevel=9, verbose=False
                )

    def test_compressed_tar_add_rebuilds_and_overwrites(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "item.txt"
            files.save(source, "one", verbose=False)
            packed = root / "items.tar.gz"
            backend = TarBackend("tar.gz")
            backend.compress(source, packed, verbose=False)

            files.save(source, "two", overwrite=True, verbose=False)
            backend.add(packed, [source], on_conflict="rename", verbose=False)
            self.assertEqual(set(backend.list_files(packed)), {"item.txt", "item_dup1.txt"})

            files.save(source, "three", overwrite=True, verbose=False)
            backend.add(packed, [source], on_conflict="overwrite", verbose=False)
            output = root / "out"
            backend.extract_file(packed, "item.txt", output)
            self.assertEqual(files.load(output / "item.txt", verbose=False), "three")


if __name__ == "__main__":
    unittest.main()

