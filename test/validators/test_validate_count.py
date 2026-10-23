"""Behavioral tests for count and positive integer validation."""

import unittest

from klygo import validators


class TestValidateCount(unittest.TestCase):
    def test_positive_integers_pass_default_minimum(self):
        self.assertIsNone(validators.validate_count(1, "batch"))
        self.assertIsNone(validators.validate_count(100, "batch"))

    def test_custom_minimum_supports_zero(self):
        self.assertIsNone(validators.validate_count(0, "warmup", minimum=0))
        self.assertIsNone(validators.validate_count(5, "warmup", minimum=0))

    def test_non_integers_and_booleans_raise_type_error(self):
        with self.assertRaisesRegex(TypeError, "batch must be int, got bool"):
            validators.validate_count(True, "batch")
        with self.assertRaisesRegex(TypeError, "batch must be int, got str"):
            validators.validate_count("1", "batch")
        with self.assertRaisesRegex(TypeError, "batch must be int, got float"):
            validators.validate_count(1.5, "batch")

    def test_values_below_minimum_raise_value_error(self):
        with self.assertRaisesRegex(ValueError, "batch must be an integer >= 1"):
            validators.validate_count(0, "batch")
        with self.assertRaisesRegex(ValueError, "warmup must be an integer >= 0"):
            validators.validate_count(-1, "warmup", minimum=0)


if __name__ == "__main__":
    unittest.main()
