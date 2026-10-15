"""Documentation and executable-example contracts for klygo.datasets."""

import inspect
import runpy
import sys
import unittest
from pathlib import Path

from klygo import datasets


class TestDocumentation(unittest.TestCase):
    def test_every_public_api_has_source_docs_markdown_and_example(self):
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
        for name in datasets.__all__:
            with self.subTest(name=name):
                self.assertTrue(inspect.getdoc(getattr(datasets, name)))
                document = repository / "docs" / "datasets" / "api" / f"{name}.md"
                example = repository / "examples" / "datasets" / f"{name}.py"
                self.assertTrue(document.is_file())
                self.assertTrue(example.is_file())
                text = document.read_text(encoding="utf-8")
                for section in required_sections:
                    self.assertIn(section, text)

    def test_every_public_example_executes(self):
        repository = Path(__file__).resolve().parents[2]
        examples = repository / "examples" / "datasets"
        sys.path.insert(0, str(examples))
        try:
            for name in datasets.__all__:
                with self.subTest(name=name):
                    runpy.run_path(
                        str(examples / f"{name}.py"),
                        run_name="__main__",
                    )
        finally:
            sys.path.remove(str(examples))

    def test_readme_documents_shared_module_boundaries(self):
        readme = Path(__file__).with_name("README.md").read_text(encoding="utf-8")
        for module in ("klygo.files", "klygo.config", "klygo.archive", "klygo.media"):
            self.assertIn(module, readme)


if __name__ == "__main__":
    unittest.main()
