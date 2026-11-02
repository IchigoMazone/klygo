"""Behavioral tests for type checking and its user-facing errors."""

import unittest

from klygo import validators


class TestValidateType(unittest.TestCase):
    def test_single_and_multiple_accepted_types(self):
        self.assertIsNone(validators.validate_type("image.jpg", str, "source"))
        self.assertIsNone(validators.validate_type(b"image", (str, bytes), "source"))

    def test_error_names_argument_expected_and_actual_types(self):
        with self.assertRaisesRegex(TypeError, "source must be str, bytes, got int"):
            validators.validate_type(42, (str, bytes), "source")


if __name__ == "__main__":
    unittest.main()
