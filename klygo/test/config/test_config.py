"""Behavioral contract for the stateful Config interface."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from klygo import Config

class TestConfig(unittest.TestCase):
    def test_complete_stateful_workflow(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manager = Config.create_default(root / "settings.yaml", {"default": {"root": str(root)}}, verbose=False)
            manager.read(verbose=False); manager.set("model.batch", 32); manager.update({"model": {"epochs": 5}}); manager.merge({"seed": 7})
            self.assertTrue(manager.has("seed")); self.assertTrue(manager.delete("model.lr"))
            self.assertEqual(json.loads(manager.to_json())["model"]["batch"], 32)
            self.assertTrue(manager.export_file("result", "toml", output_dir=root, verbose=False).is_file())
            self.assertEqual(manager.to_dict()["seed"], 7)

if __name__ == "__main__": unittest.main()
