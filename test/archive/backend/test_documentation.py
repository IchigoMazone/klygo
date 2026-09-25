"""Keep source, Markdown, examples, and exported backend symbols synchronized."""

import inspect
import unittest
from pathlib import Path

import klygo.archive.backend as backend_api


class TestBackendDocumentation(unittest.TestCase):
    def test_every_export_has_source_docs_markdown_and_example(self):
        repository = Path(__file__).resolve().parents[3]
        docs = repository / "docs" / "archive" / "backend"
        examples = repository / "examples" / "archive" / "backend"

        self.assertEqual(len(backend_api.__all__), 12)
        for name in backend_api.__all__:
            with self.subTest(symbol=name):
                symbol = getattr(backend_api, name)
                self.assertTrue(inspect.getdoc(symbol))
                self.assertTrue((docs / f"{name}.md").is_file())
                self.assertTrue((examples / f"{name}.py").is_file())

    def test_every_declared_public_backend_method_has_source_docs(self):
        backend_classes = (
            backend_api.ZipBackend,
            backend_api.TarBackend,
            backend_api.GZipBackend,
            backend_api.SevenZipBackend,
            backend_api.RarBackend,
        )
        for backend_class in backend_classes:
            for name, value in backend_class.__dict__.items():
                if name.startswith("_") or not inspect.isfunction(value):
                    continue
                with self.subTest(backend=backend_class.__name__, method=name):
                    self.assertTrue(inspect.getdoc(value))


if __name__ == "__main__":
    unittest.main()
