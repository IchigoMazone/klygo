"""Behavioral contracts for config file I/O."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from box import Box
from klygo import config, files

class TestConfigIO(unittest.TestCase):
    def test_load_save_convert_and_export(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); source = root / "settings.yaml"
            self.assertEqual(config.save(source, {"model": {"batch": 16}}, verbose=False), source)
            loaded = config.load(source, verbose=False)
            self.assertIsInstance(loaded, Box); self.assertEqual(loaded.model.batch, 16)
            converted = config.convert(source, root / "settings.toml", verbose=False)
            self.assertEqual(config.load(converted, verbose=False).model.batch, 16)
            self.assertTrue(config.export(loaded, root / "settings.json", verbose=False).is_file())
    def test_root_expansion_and_invalid_document(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); source = root / "settings.json"
            config.save(source, {"default": {"root": "/data"}, "train": {"images": "./images"}}, verbose=False)
            self.assertEqual(config.load(source, verbose=False).train.images, str(Path("/data") / "images"))
            invalid = root / "list.json"; files.save(invalid, [1, 2], verbose=False)
            with self.assertRaises(TypeError): config.load(invalid, verbose=False)

if __name__ == "__main__": unittest.main()
