"""Behavioral tests for utils.exclude_keys."""

import unittest

from klygo import utils


class TestExcludeKeys(unittest.TestCase):
    def test_excludes_one_key_without_mutating_input(self):
        source = {"threshold": 0.4, "dtype": "float32"}

        result = utils.exclude_keys(source, "dtype")

        self.assertEqual(result, {"threshold": 0.4})
        self.assertEqual(source, {"threshold": 0.4, "dtype": "float32"})

    def test_excludes_a_list_of_keys(self):
        result = utils.exclude_keys(
            {"threshold": 0.4, "dtype": "float32", "device": "cpu"},
            ["dtype", "device"],
        )

        self.assertEqual(result, {"threshold": 0.4})

    def test_rejects_unsupported_key_containers(self):
        with self.assertRaises(TypeError):
            utils.exclude_keys({"dtype": "float32"}, ("dtype",))
        with self.assertRaises(TypeError):
            utils.exclude_keys({"dtype": "float32"}, [1])


if __name__ == "__main__":
    unittest.main()
