"""Behavioral contracts for nested config access."""
import unittest
from klygo import config

class TestAccess(unittest.TestCase):
    def test_get_set_and_has_preserve_none_and_input(self):
        original = {"model": {"name": None}}
        self.assertTrue(config.has(original, "model.name")); self.assertIsNone(config.get(original, "model.name"))
        self.assertEqual(config.get(original, "missing", 4), 4)
        changed = config.set(original, "model.batch", 16)
        self.assertEqual(changed.model.batch, 16); self.assertNotIn("batch", original["model"])
    def test_delete_and_invalid_paths(self):
        data = {"model": {"batch": 16}}
        self.assertTrue(config.delete(data, "model.batch")); self.assertFalse(config.delete(data, "model.batch"))
        for function in (config.get, config.has):
            with self.assertRaises(ValueError): function(data, "")

if __name__ == "__main__": unittest.main()
