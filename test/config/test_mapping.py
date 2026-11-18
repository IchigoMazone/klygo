"""Behavioral contracts for config merge and update."""
import unittest
from klygo import config

class TestMapping(unittest.TestCase):
    def test_deep_merge_precedence_without_input_mutation(self):
        first = {"model": {"batch": 16, "epochs": 10}}
        merged = config.merge(first, {"model": {"batch": 32}}, {"seed": 7})
        self.assertEqual(merged.model.to_dict(), {"batch": 32, "epochs": 10}); self.assertEqual(first["model"]["batch"], 16)
    def test_shallow_merge_and_errors(self):
        merged = config.merge({"model": {"batch": 16, "epochs": 10}}, {"model": {"batch": 32}}, deep=False)
        self.assertEqual(merged.model.to_dict(), {"batch": 32})
        with self.assertRaises(TypeError): config.merge({}, [], deep=True)

if __name__ == "__main__": unittest.main()
