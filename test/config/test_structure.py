"""Behavioral contracts for config structure transforms."""
import unittest
from klygo import config

class TestStructure(unittest.TestCase):
    def test_round_trip_custom_separator_and_empty_mapping(self):
        nested = {"model": {"batch": 16, "options": {}}}; flat = config.flatten(nested, sep="__")
        self.assertEqual(flat, {"model__batch": 16, "model__options": {}}); self.assertEqual(config.unflatten(flat, sep="__"), nested)
    def test_conflicts_and_invalid_separator(self):
        with self.assertRaises(ValueError): config.unflatten({"a": 1, "a.b": 2})
        with self.assertRaises(ValueError): config.flatten({}, sep="")

if __name__ == "__main__": unittest.main()
