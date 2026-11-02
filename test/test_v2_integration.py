"""Offline smoke tests for the public Klygo v2 API.

These tests intentionally avoid downloading model weights, network access, and GPU
requirements so they can run in a standard CPU-only development environment.
"""

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import klygo
from klygo import config, files
from klygo.outputs import Box
from klygo.postprocessing import compose


class TestPublicPackageSurface(unittest.TestCase):
    def test_core_modules_are_exported(self):
        for name in (
            "archive",
            "config",
            "datasets",
            "files",
            "media",
            "mining",
            "outputs",
            "postprocessing",
            "preprocessing",
            "validators",
            "visual",
        ):
            with self.subTest(module=name):
                self.assertTrue(hasattr(klygo, name))

    def test_json_file_round_trip(self):
        expected = {"model": {"name": "detector", "batch": 8}, "enabled": True}
        with TemporaryDirectory() as directory:
            target = Path(directory) / "settings.json"
            files.save(target, expected, verbose=False)
            actual = files.load(target, verbose=False)
        self.assertEqual(actual, expected)

    def test_nested_config_set_get_and_has(self):
        original = {"model": {"name": "yolo"}}
        updated = config.set(original, "model.batch", 16)
        self.assertEqual(config.get(updated, "model.batch"), 16)
        self.assertTrue(config.has(updated, "model.name"))
        self.assertFalse(config.has(updated, "model.missing"))
        # Functional set must not mutate its input.
        self.assertNotIn("batch", original["model"])

    def test_box_rejects_invalid_confidence(self):
        with self.assertRaises(ValueError):
            Box((0, 0, 10, 10), label="car", score=1.5)

    def test_box_rejects_degenerate_coordinates(self):
        with self.assertRaises(ValueError):
            Box((5, 5, 5, 10), label="car", score=0.9)

    def test_postprocessing_pipeline_preserves_order(self):
        result = []
        pipeline = compose(
            lambda values: values.append("first"),
            lambda values: values.append("second"),
        )
        pipeline(result)
        self.assertEqual(result, ["first", "second"])


if __name__ == "__main__":
    unittest.main()
