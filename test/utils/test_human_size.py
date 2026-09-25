"""Behavioral contract for klygo.utils.human_size."""

import unittest

from klygo import utils


class TestHumanSize(unittest.TestCase):
    def test_units_precision_and_validation(self):
        self.assertEqual(utils.human_size(0), "0.00 B")
        self.assertEqual(utils.human_size(1024), "1.00 KB")
        self.assertEqual(utils.human_size(1024 ** 2, 1), "1.0 MB")
        with self.assertRaises(TypeError):
            utils.human_size(1.5)
        with self.assertRaises(ValueError):
            utils.human_size(-1)


if __name__ == "__main__":
    unittest.main()
