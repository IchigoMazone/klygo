"""Behavioral contracts for config defaults and creation."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from klygo import config

class TestCreation(unittest.TestCase):
    def test_defaults_are_independent_and_support_deep_overrides(self):
        first = config.defaults({"model": {"batch": 64}}); second = config.defaults()
        first["model"]["name"] = "changed"
        self.assertEqual(first["model"]["batch"], 64); self.assertEqual(second["model"]["name"], "yolov8n")
    def test_create_round_trip_and_overwrite_policy(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "settings.yaml"
            self.assertEqual(config.create(path, {"model": {"epochs": 5}}, verbose=False).model.epochs, 5)
            with self.assertRaises(FileExistsError): config.create(path, verbose=False)

if __name__ == "__main__": unittest.main()
