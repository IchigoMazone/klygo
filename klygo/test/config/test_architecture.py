"""Package layout and public export contract for klygo.config."""

import unittest
from pathlib import Path

from klygo import config


class TestArchitecture(unittest.TestCase):
    def test_grouped_implementation_modules_replace_monolith(self):
        package = Path(config.__file__).parent
        self.assertFalse((package / "operations.py").exists())
        for module in (
            "access.py", "creation.py", "environment.py", "inspection.py",
            "io.py", "mapping.py", "structure.py", "validation.py",
        ):
            with self.subTest(module=module):
                self.assertTrue((package / module).is_file())

    def test_public_exports_are_unique_and_available(self):
        self.assertEqual(len(config.__all__), 21)
        self.assertEqual(len(config.__all__), len(set(config.__all__)))
        for name in config.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(config, name))


if __name__ == "__main__":
    unittest.main()
