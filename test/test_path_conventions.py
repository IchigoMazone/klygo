"""Keep runtime path conversion in the shared files package."""

import ast
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "klygo"
PATH_CONSUMERS = (
    "archive",
    "config",
    "datasets",
    "media",
    "validators/datasets",
)


class TestPathConventions(unittest.TestCase):
    def test_path_consumers_use_files_for_runtime_construction(self):
        for package in PATH_CONSUMERS:
            for source in (PACKAGE_ROOT / package).rglob("*.py"):
                with self.subTest(source=source.relative_to(PACKAGE_ROOT)):
                    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
                    constructors = set()
                    pathlib_modules = set()
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ImportFrom) and node.module == "pathlib":
                            constructors.update(
                                alias.asname or alias.name
                                for alias in node.names
                                if alias.name == "Path"
                            )
                        elif isinstance(node, ast.Import):
                            pathlib_modules.update(
                                alias.asname or alias.name
                                for alias in node.names
                                if alias.name == "pathlib"
                            )
                    violations = []
                    for node in ast.walk(tree):
                        if not isinstance(node, ast.Call):
                            continue
                        func = node.func
                        direct = isinstance(func, ast.Name) and func.id in constructors
                        qualified = (
                            isinstance(func, ast.Attribute)
                            and func.attr == "Path"
                            and isinstance(func.value, ast.Name)
                            and func.value.id in pathlib_modules
                        )
                        path_type_argument = any(
                            isinstance(child, ast.Name) and child.id in constructors
                            for argument in (*node.args, *(keyword.value for keyword in node.keywords))
                            for child in ast.walk(argument)
                        )
                        if direct or qualified or path_type_argument:
                            violations.append(node.lineno)
                    self.assertEqual(
                        violations,
                        [],
                        f"{source}: use files.path(...) for runtime paths",
                    )

    def test_path_consumers_use_files_for_filesystem_operations(self):
        direct_methods = {
            "exists", "is_file", "mkdir", "rglob", "glob", "iterdir",
            "rename", "resolve", "stat", "unlink", "is_symlink",
            "relative_to", "with_suffix",
        }
        for package in PATH_CONSUMERS:
            for source in (PACKAGE_ROOT / package).rglob("*.py"):
                with self.subTest(source=source.relative_to(PACKAGE_ROOT)):
                    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
                    forbidden = direct_methods.copy()
                    if "backend" not in source.parts:
                        forbidden.update({"is_dir", "as_posix"})
                    violations = [
                        node.lineno
                        for node in ast.walk(tree)
                        if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr in forbidden
                        and not (
                            isinstance(node.func.value, ast.Name)
                            and node.func.value.id in {"files", "file_utils", "fs"}
                        )
                    ]
                    self.assertEqual(
                        violations,
                        [],
                        f"{source}: use the equivalent klygo.files API",
                    )


if __name__ == "__main__":
    unittest.main()
