"""Package layout and public export contract for klygo.config."""

import importlib
import unittest
from pathlib import Path

from klygo import config


class TestArchitecture(unittest.TestCase):
    def test_grouped_implementation_modules_replace_monolith(self):
        package = Path(config.__file__).parent
        self.assertFalse((package / "operations.py").exists())
        submodules = (
            "access", "config", "creation", "environment", "inspection",
            "io", "mapping", "structure", "validation",
        )
        combined: list[str] = []
        for stem in submodules:
            with self.subTest(module=f"{stem}.py"):
                self.assertTrue((package / f"{stem}.py").is_file())
                mod = importlib.import_module(f"klygo.config.{stem}")
                self.assertTrue(hasattr(mod, "__all__"))
                combined.extend(mod.__all__)
        self.assertEqual(set(combined), set(config.__all__))
        self.assertEqual(len(combined), len(config.__all__))

    def test_public_exports_are_unique_and_available(self):
        self.assertEqual(len(config.__all__), 21)
        self.assertEqual(len(config.__all__), len(set(config.__all__)))
        for name in config.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(config, name))


if __name__ == "__main__":
    unittest.main()
