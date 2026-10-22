"""Behavioral tests for utils.invert."""

import unittest

from klygo import utils


class TestInvert(unittest.TestCase):
    def test_swaps_keys_and_values_in_both_directions(self):
        source = {"cat": 0, "dog": 1}

        swapped = utils.invert(source)
        restored = utils.invert(swapped)

        self.assertEqual(swapped, {0: "cat", 1: "dog"})
        self.assertEqual(restored, source)
        self.assertEqual(source, {"cat": 0, "dog": 1})

    def test_rejects_duplicate_values(self):
        with self.assertRaises(ValueError):
            utils.invert({"cat": 0, "dog": 0})

    def test_rejects_unhashable_values(self):
        with self.assertRaises(TypeError):
            utils.invert({"classes": ["cat", "dog"]})

    def test_rejects_non_mapping_input(self):
        with self.assertRaises(TypeError):
            utils.invert([("cat", 0)])


if __name__ == "__main__":
    unittest.main()
