"""Behavioral contracts for environment overlays."""
import os
import unittest
from unittest.mock import patch
from klygo import config

class TestEnvironment(unittest.TestCase):
    def test_string_and_parsed_values(self):
        values = {"APP_MODEL_BATCH": "32", "APP_DEBUG": "true", "UNRELATED": "x"}
        with patch.dict(os.environ, values, clear=True):
            strings = config.from_env({"model": {"name": "yolo"}}, prefix="APP_")
            parsed = config.from_env(prefix="APP_", parse_values=True)
        self.assertEqual(strings["model"]["batch"], "32"); self.assertEqual(parsed["model"]["batch"], 32)
        self.assertIs(parsed["debug"], True)

if __name__ == "__main__": unittest.main()
