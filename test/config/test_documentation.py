"""Documentation and example coverage contract for klygo.config."""

import inspect
import runpy
import unittest
from pathlib import Path

from klygo import config


FUNCTIONS = [name for name in config.__all__ if name != "Config"]


class TestDocumentation(unittest.TestCase):
    def test_every_export_has_source_docs_markdown_and_example(self):
        repository = Path(__file__).resolve().parents[2]
        required_sections = (
            "## Contract",
            "## Parameters",
            "## Returns",
            "## Errors and edge cases",
            "## AI usage guidance",
            "## Example",
            "## Tests",
            "## Complete executable example",
        )
        for name in config.__all__:
            with self.subTest(name=name):
                value = getattr(config, name)
                self.assertTrue(inspect.getdoc(value))
                document = repository / "docs" / "config" / f"{name}.md"
                self.assertTrue(document.is_file())
                self.assertTrue((repository / "examples" / "config" / f"{name}.py").is_file())
                text = document.read_text(encoding="utf-8")
                for section in required_sections:
                    self.assertIn(section, text)

    def test_every_example_executes(self):
        examples = Path(__file__).resolve().parents[2] / "examples" / "config"
        for name in config.__all__:
            with self.subTest(name=name):
                runpy.run_path(str(examples / f"{name}.py"), run_name="__main__")

    def test_config_public_methods_have_docstrings(self):
        methods_with_parameters = {
            "read", "to_json", "get", "set", "has", "delete", "merge",
            "update", "create_default", "export_file",
        }
        for name in (
            "config_path", "read", "to_dict", "to_json", "get", "set",
            "has", "delete", "merge", "update", "create_default",
            "export_file",
        ):
            with self.subTest(name=name):
                value = inspect.getattr_static(config.Config, name)
                if isinstance(value, property):
                    value = value.fget
                elif isinstance(value, classmethod):
                    value = value.__func__
                doc = inspect.getdoc(value) or ""
                self.assertIn("Returns\n", doc)
                self.assertIn("Examples\n", doc)
                if name in methods_with_parameters:
                    self.assertIn("Parameters\n", doc)


if __name__ == "__main__":
    unittest.main()
