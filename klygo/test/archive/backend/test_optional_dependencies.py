"""Packaging contract for optional archive dependencies."""

import tomllib
import unittest
from pathlib import Path


class TestOptionalDependencies(unittest.TestCase):
    def test_archive_packages_have_independent_install_extras(self):
        repository = Path(__file__).resolve().parents[3]
        with open(repository / "pyproject.toml", "rb") as project_file:
            project = tomllib.load(project_file)

        optional = project["project"]["optional-dependencies"]
        self.assertEqual(optional["py7zr"], ["py7zr"])
        self.assertEqual(optional["rarfile"], ["rarfile"])

    def test_docs_show_pip_and_both_uv_install_modes(self):
        repository = Path(__file__).resolve().parents[3]
        documents = (
            repository / "README.md",
            repository / "docs" / "archive" / "backend" / "README.md",
            repository / "docs" / "archive" / "backend" / "SevenZipBackend.md",
            repository / "docs" / "archive" / "backend" / "RarBackend.md",
        )
        for document in documents:
            text = document.read_text(encoding="utf-8")
            with self.subTest(document=document.name):
                self.assertIn("pip install", text)
                self.assertIn("uv add", text)
                self.assertIn("uv pip install", text)


if __name__ == "__main__":
    unittest.main()
