"""Documentation, example, and tutorial coverage contract for klygo.files."""

import importlib
import inspect
import io
import re
import runpy
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from urllib.parse import unquote, urlsplit

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
                source_doc = inspect.getdoc(value)
                self.assertTrue(source_doc)
                for heading in ("Parameters", "Returns", "Raises", "Examples"):
                    self.assertIn(f"{heading}\n{'-' * len(heading)}", source_doc)
                document = docs / f"{name}.md"
                self.assertTrue(document.is_file(), f"Missing doc file: {document}")
                example = examples / f"{name}.py"
                self.assertTrue(example.is_file(), f"Missing example file: {example}")
                self.assertNotIn("from pathlib import Path", example.read_text(encoding="utf-8"))
                text = document.read_text(encoding="utf-8")
                for section in required_sections:
                    self.assertIn(section, text, f"Missing section '{section}' in {document.name}")

    def test_every_example_executes(self):
        examples = Path(__file__).resolve().parents[2] / "examples" / "files"
        for name in files.__all__:
            with self.subTest(name=name):
                with redirect_stdout(io.StringIO()):
                    runpy.run_path(str(examples / f"{name}.py"), run_name="__main__")

    def test_complete_examples_match_runnable_sources(self):
        repository = Path(__file__).resolve().parents[2]
        for name in files.__all__:
            with self.subTest(name=name):
                document = repository / "docs" / "files" / "api" / f"{name}.md"
                text = document.read_text(encoding="utf-8").split("## Complete executable example", 1)[1]
                code = re.search(r"```python\n(.*?)\n```", text, flags=re.DOTALL)
                self.assertIsNotNone(code)
                example = repository / "examples" / "files" / f"{name}.py"
                self.assertEqual(code.group(1).strip(), example.read_text(encoding="utf-8").strip())

    def test_submodules_and_tutorial_notebooks_are_synchronized(self):
        repository = Path(__file__).resolve().parents[2]
        tutorials = repository / "tutorials" / "files"
        expected_map = {
            "Io": importlib.import_module("klygo.files.io").__all__,
            "Download": importlib.import_module("klygo.files.download").__all__,
            "Filesystem": importlib.import_module("klygo.files.filesystem").__all__,
            "Metadata": importlib.import_module("klygo.files.metadata").__all__,
            "Paths": importlib.import_module("klygo.files.paths").__all__,
        }
        combined = [name for group in expected_map.values() for name in group]
        self.assertEqual(set(combined), set(files.__all__))
        self.assertEqual(len(combined), len(files.__all__))

        overview = tutorials / "Document_Files_Overview.ipynb"
        self.assertTrue(overview.is_file())
        overview_text = overview.read_text(encoding="utf-8")
        for heading in ("## Choose a guide", "## Remember"):
            self.assertIn(heading, overview_text, f"Missing '{heading}' in {overview.name}")

        actual_notebooks = {p.name for p in tutorials.glob("*.ipynb")}
        expected_notebooks = {"Document_Files_Overview.ipynb"} | {
            f"Document_Files_{suffix}.ipynb" for suffix in expected_map
        }
        self.assertEqual(actual_notebooks, expected_notebooks)

        for suffix, api_names in expected_map.items():
            notebook = tutorials / f"Document_Files_{suffix}.ipynb"
            text = notebook.read_text(encoding="utf-8")
            for heading in ("## Jump to an API", "## Put it together", "## Remember"):
                self.assertIn(heading, text, f"Missing '{heading}' in {notebook.name}")
            for api_name in api_names:
                with self.subTest(notebook=notebook.name, api=api_name):
                    self.assertIn(f"files.{api_name}", text)

    def test_native_pathlib_guide_examples_execute(self):
        repository = Path(__file__).resolve().parents[2]
        guide = repository / "docs" / "files" / "pathlib.md"
        text = guide.read_text(encoding="utf-8")
        blocks = re.findall(r"```python\n(.*?)\n```", text, flags=re.DOTALL)
        self.assertGreaterEqual(len(blocks), 3)
        example = repository / "examples" / "files" / "native_pathlib.py"
        self.assertEqual(blocks[-1].strip(), example.read_text(encoding="utf-8").strip())
        for index, code in enumerate(blocks):
            with self.subTest(block=index), redirect_stdout(io.StringIO()):
                exec(compile(code, str(guide), "exec"), {"__name__": "__main__"})

    def test_markdown_local_links_and_fences(self):
        docs = Path(__file__).resolve().parent
        for document in docs.rglob("*.md"):
            with self.subTest(document=document.name):
                text = document.read_text(encoding="utf-8")
                fences = re.findall(r"^```[^`\r\n]*$", text, flags=re.MULTILINE)
                self.assertEqual(len(fences) % 2, 0, f"Unclosed code fence in {document}")
                for target in re.findall(r"\[[^\]]+\]\(([^)\s]+)\)", text):
                    url = urlsplit(target)
                    if url.scheme or url.netloc or not url.path:
                        continue
                    resolved = document.parent / unquote(url.path)
                    self.assertTrue(resolved.exists(), f"Broken link in {document}: {target}")


if __name__ == "__main__":
    unittest.main()
