"""Behavioral contracts for config validation."""
import unittest
from klygo import config

class TestValidation(unittest.TestCase):
    def test_paths_types_predicates_and_literals(self):
        data = {"model": {"batch": 16, "name": "yolo"}}
        self.assertTrue(config.validate(data, ["model.batch"]))
        self.assertTrue(config.validate(data, {"model.batch": int, "model.name": lambda value: bool(value)}))
        self.assertTrue(config.validate(data, {"model.name": "yolo"}))
        with self.assertRaises(ValueError): config.validate(data, {"model.batch": str})
        with self.assertRaises(ValueError): config.validate(data, ["model.missing"])

if __name__ == "__main__": unittest.main()
