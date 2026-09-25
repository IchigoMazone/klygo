"""Documentation and example coverage contract for klygo.files."""

import inspect
import runpy
import unittest
from pathlib import Path

from klygo import files


class TestFilesDocumentation(unittest.TestCase):
    def test_every_export_has_source_docs_markdown_and_example(self):
        repository = Path(__file__).resolve().parents[2]
        docs = repository / "docs" / "files" / "api"
        examples = repository / "examples" / "files"
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
        for name in files.__all__:
            with self.subTest(name=name):
                value = getattr(files, name)
                self.assertTrue(inspect.getdoc(value))
                document = docs / f"{name}.md"
                self.assertTrue(document.is_file(), f"Missing doc file: {document}")
                self.assertTrue((examples / f"{name}.py").is_file(), f"Missing example file: {examples / f'{name}.py'}")
                text = document.read_text(encoding="utf-8")
                for section in required_sections:
                    self.assertIn(section, text, f"Missing section '{section}' in {document.name}")

    def test_every_example_executes(self):
        examples = Path(__file__).resolve().parents[2] / "examples" / "files"
        for name in files.__all__:
            with self.subTest(name=name):
                runpy.run_path(str(examples / f"{name}.py"), run_name="__main__")


if __name__ == "__main__":
    unittest.main()
