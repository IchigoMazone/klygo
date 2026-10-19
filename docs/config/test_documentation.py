"""Documentation, example, and tutorial coverage contract for klygo.config."""

import importlib
import inspect
import re
import runpy
import unittest
from pathlib import Path
from urllib.parse import unquote, urlsplit

from klygo import config


FUNCTIONS = [name for name in config.__all__ if name != "Config"]


class TestDocumentation(unittest.TestCase):
    def test_config_methods_use_public_docstring_form(self):
        for name, method in inspect.getmembers(config.Config, inspect.isfunction):
            if name.startswith("_"):
                continue
            with self.subTest(name=name):
                source_doc = inspect.getdoc(method)
                for heading in ("Parameters", "Returns", "Raises", "Examples"):
                    self.assertIn(f"{heading}\n{'-' * len(heading)}", source_doc)

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
                source_doc = inspect.getdoc(value)
                self.assertTrue(source_doc)
                for heading in ("Parameters", "Returns", "Raises", "Examples"):
                    self.assertIn(f"{heading}\n{'-' * len(heading)}", source_doc)
                document = repository / "docs" / "config" / "api" / f"{name}.md"
                self.assertTrue(document.is_file())
                example = repository / "examples" / "config" / f"{name}.py"
                self.assertTrue(example.is_file())
                self.assertNotIn("from pathlib import Path", example.read_text(encoding="utf-8"))
                text = document.read_text(encoding="utf-8")
                for section in required_sections:
                    self.assertIn(section, text)

    def test_every_example_executes(self):
        examples = Path(__file__).resolve().parents[2] / "examples" / "config"
        for name in config.__all__:
            with self.subTest(name=name):
                runpy.run_path(str(examples / f"{name}.py"), run_name="__main__")

    def test_submodules_and_tutorial_notebooks_are_synchronized(self):
        repository = Path(__file__).resolve().parents[2]
        tutorials = repository / "tutorials" / "config"
        expected_map = {
            "Io": importlib.import_module("klygo.config.io").__all__,
            "Creation": importlib.import_module("klygo.config.creation").__all__,
            "Mapping": importlib.import_module("klygo.config.mapping").__all__,
            "Access": importlib.import_module("klygo.config.access").__all__,
            "Inspection": importlib.import_module("klygo.config.inspection").__all__,
            "Structure": importlib.import_module("klygo.config.structure").__all__,
            "Environment": importlib.import_module("klygo.config.environment").__all__,
            "Validation": importlib.import_module("klygo.config.validation").__all__,
            "Config": importlib.import_module("klygo.config.config").__all__,
        }
        combined = [name for group in expected_map.values() for name in group]
        self.assertEqual(set(combined), set(config.__all__))
        self.assertEqual(len(combined), len(config.__all__))

        overview = tutorials / "Document_Config_Overview.ipynb"
        self.assertTrue(overview.is_file())
        overview_text = overview.read_text(encoding="utf-8")
        for heading in ("## Choose a guide", "## Remember"):
            self.assertIn(heading, overview_text, f"Missing '{heading}' in {overview.name}")

        actual_notebooks = {p.name for p in tutorials.glob("*.ipynb")}
        expected_notebooks = {"Document_Config_Overview.ipynb"} | {
            f"Document_Config_{suffix}.ipynb" for suffix in expected_map
        }
        self.assertEqual(actual_notebooks, expected_notebooks)

        for suffix, api_names in expected_map.items():
            notebook = tutorials / f"Document_Config_{suffix}.ipynb"
            text = notebook.read_text(encoding="utf-8")
            for heading in ("## Jump to an API", "## Put it together", "## Remember"):
                self.assertIn(heading, text, f"Missing '{heading}' in {notebook.name}")
            for api_name in api_names:
                with self.subTest(notebook=notebook.name, api=api_name):
                    self.assertTrue(
                        f"config.{api_name}" in text or api_name in text,
                        f"Missing {api_name} in {notebook.name}",
                    )

    def test_model_metadata_guide_and_example_stay_executable(self):
        repository = Path(__file__).resolve().parents[2]
        guide = repository / "docs" / "config" / "model-metadata.md"
        example = repository / "examples" / "config" / "model_metadata.py"

        self.assertTrue(guide.is_file())
        self.assertTrue(example.is_file())
        text = guide.read_text(encoding="utf-8")
        for public_api in (
            "models.flags",
            "models.metadata",
            "models.configure",
            "Config.create_default",
            "config.save",
            "config.load",
        ):
            with self.subTest(public_api=public_api):
                self.assertIn(public_api, text)

        runpy.run_path(str(example), run_name="__main__")

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

    def test_markdown_local_links_resolve(self):
        docs_root = Path(__file__).resolve().parent
        link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
        for document in docs_root.rglob("*.md"):
            text = document.read_text(encoding="utf-8")
            for raw_target in link_pattern.findall(text):
                parsed = urlsplit(raw_target)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                target = (document.parent / unquote(parsed.path)).resolve()
                with self.subTest(document=document.name, target=raw_target):
                    self.assertTrue(target.exists(), f"Broken link '{raw_target}' in {document}")


if __name__ == "__main__":
    unittest.main()
