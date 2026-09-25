"""Behavioral contracts for config views and comparison."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from klygo import config

class TestInspection(unittest.TestCase):
    def test_flat_and_top_level_views(self):
        data = {"a": 1, "b": {"c": 2}}
        self.assertEqual(config.keys(data), ["a", "b"]); self.assertEqual(config.values(data, flat=True), [1, 2])
        self.assertEqual(dict(config.items(data, flat=True)), {"a": 1, "b.c": 2})
    def test_diff_objects_and_files(self):
        expected = {"added": {"b": 3}, "removed": {}, "modified": {"a": {"from": 1, "to": 2}}}
        self.assertEqual(config.diff({"a": 1}, {"a": 2, "b": 3}), expected)
        with TemporaryDirectory() as directory:
            first, second = Path(directory) / "a.json", Path(directory) / "b.json"
            config.save(first, {"a": 1}, verbose=False); config.save(second, {"a": 2, "b": 3}, verbose=False)
            self.assertEqual(config.diff(first, second), expected)

if __name__ == "__main__": unittest.main()
