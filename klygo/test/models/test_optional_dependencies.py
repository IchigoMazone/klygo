"""Packaging and lazy-import contracts for optional model frameworks."""

import subprocess
import sys
import tomllib
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]


class TestOptionalModelDependencies(unittest.TestCase):
    def test_model_frameworks_are_not_core_dependencies(self):
        with open(REPOSITORY / "pyproject.toml", "rb") as project_file:
            project = tomllib.load(project_file)["project"]

        dependencies = {item.split("<", 1)[0].split(">", 1)[0] for item in project["dependencies"]}
        self.assertNotIn("torch", dependencies)
        self.assertNotIn("transformers", dependencies)
        self.assertNotIn("ultralytics", dependencies)
        self.assertNotIn("keras-hub", dependencies)

    def test_each_model_stack_has_an_install_extra(self):
        with open(REPOSITORY / "pyproject.toml", "rb") as project_file:
            optional = tomllib.load(project_file)["project"]["optional-dependencies"]

        self.assertEqual(optional["torch"], ["torch>=2.7,<3.0"])
        self.assertEqual(
            optional["transformers"],
            ["torch>=2.7,<3.0", "transformers>=4.50,<6.0"],
        )
        self.assertEqual(optional["ultralytics"], ["ultralytics"])
        self.assertEqual(optional["keras-hub"], ["keras-hub"])
        self.assertEqual(
            set(optional["models"]),
            {"torch>=2.7,<3.0", "transformers>=4.50,<6.0", "ultralytics", "keras-hub"},
        )

    def test_core_and_models_import_without_optional_frameworks(self):
        script = r'''
import builtins
import sys

blocked_roots = {
    "jax", "keras", "keras_hub", "tensorflow", "torch", "transformers", "ultralytics"
}
real_import = builtins.__import__

def block_optional(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] in blocked_roots:
        raise ImportError(f"blocked optional dependency: {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = block_optional
import klygo
import klygo.models

assert not blocked_roots.intersection(sys.modules)
'''
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_importing_models_does_not_eagerly_load_installed_frameworks(self):
        script = r'''
import sys
import klygo.models

frameworks = {"jax", "keras", "tensorflow", "torch", "transformers", "ultralytics"}
assert not frameworks.intersection(sys.modules), frameworks.intersection(sys.modules)
'''
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_missing_model_extras_raise_actionable_errors(self):
        script = r'''
import builtins
from klygo.models.backend import common

blocked_roots = {"jax", "keras", "tensorflow", "torch"}
real_import = builtins.__import__

def block_optional(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] in blocked_roots:
        raise ImportError(f"blocked optional dependency: {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = block_optional
for backend, extra in (
    ("Hugging Face", "klygo[transformers]"),
    ("Ultralytics", "klygo[ultralytics]"),
    ("KerasHub", "klygo[keras-hub]"),
):
    try:
        common.current_device(None, backend=backend)
    except ImportError as exc:
        assert extra in str(exc), (backend, str(exc))
    else:
        raise AssertionError(f"{backend} did not report its missing extra")
'''
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_readme_documents_pip_and_uv_for_model_extras(self):
        readme = (REPOSITORY / "README.md").read_text(encoding="utf-8")
        for extra in ("transformers", "ultralytics", "keras-hub", "models"):
            with self.subTest(extra=extra):
                self.assertIn(f'pip install "klygo[{extra}]"', readme)
                self.assertIn(f'uv add "klygo[{extra}]"', readme)
                self.assertIn(f'uv pip install "klygo[{extra}]"', readme)


if __name__ == "__main__":
    unittest.main()
