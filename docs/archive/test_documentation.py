"""Documentation and example coverage contract for klygo.archive."""

import inspect
import runpy
import unittest
from pathlib import Path

from klygo import archive


class TestArchiveDocumentation(unittest.TestCase):
    def test_every_export_has_source_docs_markdown_and_example(self):
        repository = Path(__file__).resolve().parents[2]
        docs = repository / "docs" / "archive" / "api"
        examples = repository / "examples" / "archive"
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
        for name in archive.__all__:
            with self.subTest(name=name):
                value = getattr(archive, name)
                source_doc = inspect.getdoc(value)
                self.assertTrue(source_doc)
                for heading in ("Parameters", "Returns", "Raises", "Examples"):
                    self.assertIn(f"{heading}\n{'-' * len(heading)}", source_doc)
                document = docs / f"{name}.md"
                self.assertTrue(document.is_file(), f"Missing doc file: {document}")
                self.assertTrue((examples / f"{name}.py").is_file(), f"Missing example file: {examples / f'{name}.py'}")
                text = document.read_text(encoding="utf-8")
                for section in required_sections:
                    self.assertIn(section, text, f"Missing section '{section}' in {document.name}")

    def test_every_example_executes(self):
        examples = Path(__file__).resolve().parents[2] / "examples" / "archive"
        for name in archive.__all__:
            with self.subTest(name=name):
                runpy.run_path(str(examples / f"{name}.py"), run_name="__main__")

    def test_archive_file_methods_use_public_docstring_form(self):
        for name, method in inspect.getmembers(archive.ArchiveFile, inspect.isfunction):
            if name.startswith("_"):
                continue
            with self.subTest(name=name):
                source_doc = inspect.getdoc(method)
                for heading in ("Parameters", "Returns", "Raises", "Examples"):
                    self.assertIn(f"{heading}\n{'-' * len(heading)}", source_doc)


if __name__ == "__main__":
    unittest.main()
