"""Audit public API coverage, 1-to-1 submodule alignment, Colab links, and cross-module dogfooding across Klygo."""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import io
import re
import runpy
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from urllib.parse import unquote, urlsplit

ALL_MODULES = [
    "files",
    "archive",
    "config",
    "datasets",
    "media",
    "models",
    "outputs",
    "postprocessing",
    "preprocessing",
    "utils",
    "validators",
    "visual",
]

COLAB_BASE = "https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials"


def _contains_non_english(text: str) -> bool:
    """Return True if text contains Vietnamese diacritics or non-Latin alphabets."""
    for c in text:
        code = ord(c)
        if 0x1EA0 <= code <= 0x1EF9:
            return True
        if code in (0x102, 0x103, 0x110, 0x111, 0x128, 0x129, 0x168, 0x169, 0x1A0, 0x1A1, 0x1AF, 0x1B0):
            return True
        if 0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF or 0xAC00 <= code <= 0xD7AF:
            return True
    return False


def _title_stem(stem: str) -> str:
    return "_".join(part.capitalize() for part in stem.split("_"))


def sync_colab_in_notebook(nb_path: Path, tut_dir: Path) -> bool:
    """Synchronize Colab badge and navigation links inside a notebook."""
    import json

    rel = nb_path.relative_to(tut_dir).as_posix()
    parts = rel.split("/")
    module = parts[0]
    nb_name = nb_path.name
    expected_url = f"{COLAB_BASE}/{module}/{nb_name}"
    badge_md = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({expected_url})"

    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception:
        return False

    changed = False

    # 1. First markdown cell: ensure Colab badge
    first_md = None
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            first_md = cell
            break

    if first_md:
        src = "".join(first_md.get("source", []))
        if "assets/colab-badge.svg" not in src:
            lines = list(first_md.get("source", []))
            title_idx = -1
            for idx, line in enumerate(lines):
                if line.startswith("#"):
                    title_idx = idx
                    break
            if title_idx >= 0:
                lines.insert(title_idx + 1, f"\n{badge_md}\n\n")
            else:
                lines.insert(0, f"{badge_md}\n\n")
            first_md["source"] = lines
            changed = True
        else:
            pattern = r"\[!\[Open In Colab\]\(https://colab\.research\.google\.com/assets/colab-badge\.svg\)\]\([^)]+\)"
            new_src = re.sub(pattern, badge_md, src)
            if new_src != src:
                first_md["source"] = [new_src]
                changed = True

    # 2. Update table & navigation Colab URLs
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            src = "".join(cell.get("source", []))

            def repl_nav(match):
                doc_name = match.group(1)
                colab_link = f"{COLAB_BASE}/{module}/{doc_name}"
                return f"[{doc_name}]({doc_name}) ([Colab]({colab_link}))"

            new_src = re.sub(
                r"\[(Document_[A-Za-z0-9_]+\.ipynb)\]\(\1\)\s*\(\[Colab\]\([^)]+\)\)",
                repl_nav,
                src,
            )

            def repl_table(match):
                doc_name = match.group(1)
                colab_link = f"{COLAB_BASE}/{module}/{doc_name}"
                return f"[{doc_name}]({doc_name}) | [Colab]({colab_link})"

            new_src = re.sub(
                r"\[(Document_[A-Za-z0-9_]+\.ipynb)\]\(\1\)\s*\|\s*\[(?:Colab|Open in Colab)\]\([^)]+\)",
                repl_table,
                new_src,
            )

            if new_src != src:
                cell["source"] = [new_src]
                changed = True

    if changed:
        with open(nb_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
        return True
    return False


def _discover_public_submodules(module_name: str, module_dir: Path, exports: set[str]) -> dict[str, list[str]]:
    """Return mapping of public submodule stem -> exported API names belonging to that submodule."""
    submodules: dict[str, list[str]] = {}
    candidates = sorted(
        p for p in module_dir.glob("*.py") if not p.name.startswith("_")
    )
    for path in candidates:
        sub = importlib.import_module(f"klygo.{module_name}.{path.stem}")
        sub_all = list(getattr(sub, "__all__", ()))
        if not sub_all:
            sub_all = [
                name
                for name in exports
                if getattr(getattr(sub, name, None), "__module__", "") == sub.__name__
            ]
        if set(sub_all) == exports and len(candidates) > 1:
            continue
        matched = [name for name in sub_all if name in exports]
        if matched:
            submodules[path.stem] = matched
    return submodules


def audit_module(module_name: str, root: Path, run_examples: bool = True) -> tuple[int, int, list[str]]:
    """Audit a single Klygo module. Returns (export_count, submodule_count, failures)."""
    failures: list[str] = []

    sys.path.insert(0, str(root))
    try:
        public_module = importlib.import_module(f"klygo.{module_name}")
    except Exception as exc:
        return 0, 0, [f"failed to import klygo.{module_name}: {exc}"]

    exports = list(getattr(public_module, "__all__", ()))
    if not exports:
        return 0, 0, [f"klygo.{module_name} has no public __all__ inventory"]

    module_dir = root / "klygo" / module_name
    docs_dir = root / "docs" / module_name
    api_docs = docs_dir / "api"
    examples = root / "examples" / module_name
    tutorials = root / "tutorials" / module_name

    notebook_text = ""
    for notebook in sorted(tutorials.glob("*.ipynb")) if tutorials.exists() else ():
        notebook_text += notebook.read_text(encoding="utf-8")

    # 1. API docstrings, markdown docs, and examples
    base_sys_path = [str(root)] + [p for p in sys.path if p != str(root) and "examples" not in p]

    for name in exports:
        value = getattr(public_module, name, None)
        if value is None:
            failures.append(f"missing exported attribute: {name}")
            continue
        doc = inspect.getdoc(value)
        if not doc:
            failures.append(f"missing public docstring: {name}")
        elif _contains_non_english(doc):
            failures.append(f"public docstring must be in English: klygo.{module_name}.{name}")

        doc_file = api_docs / f"{name}.md"
        # Support case-insensitive matching for Windows/Linux consistency
        if not doc_file.is_file() and not any(p.name.lower() == f"{name.lower()}.md" for p in api_docs.glob("*.md")):
            failures.append(f"missing API doc: docs/{module_name}/api/{name}.md")

        example_file = examples / f"{name}.py"
        if not example_file.is_file() and not any(p.name.lower() == f"{name.lower()}.py" for p in examples.glob("*.py")):
            failures.append(f"missing example: examples/{module_name}/{name}.py")
        else:
            actual_example = example_file if example_file.is_file() else next(p for p in examples.glob("*.py") if p.name.lower() == f"{name.lower()}.py")
            ex_text = actual_example.read_text(encoding="utf-8")
            if module_name != "files" or name != "native_pathlib":
                if re.search(r"^\s*from\s+pathlib\s+import\s+Path\b", ex_text, flags=re.MULTILINE):
                    failures.append(
                        f"example imports pathlib.Path directly instead of using klygo.files.path(): examples/{module_name}/{actual_example.name}"
                    )
            # Execute example if requested
            if run_examples:
                buf = io.StringIO()
                if "datasets" in str(actual_example):
                    sys.path = [str(actual_example.parent)] + list(base_sys_path)
                else:
                    sys.path = list(base_sys_path)
                try:
                    with redirect_stdout(buf), redirect_stderr(buf):
                        runpy.run_path(str(actual_example), run_name="__main__")
                except Exception as exc:
                    failures.append(f"example failed to execute ({actual_example.name}): {type(exc).__name__}: {exc}")

        if notebook_text and not re.search(rf"\b{re.escape(name)}\b", notebook_text):
            failures.append(f"missing tutorial mention: {name}")

    # 2. Submodules and tutorial notebooks alignment
    submodules = _discover_public_submodules(module_name, module_dir, set(exports))
    mod_title = _title_stem(module_name)
    if tutorials.exists() and submodules:
        expected_notebooks = {f"Document_{mod_title}_Overview.ipynb"} | {
            f"Document_{mod_title}_{_title_stem(stem)}.ipynb" for stem in submodules
        }
        actual_notebooks = {p.name for p in tutorials.glob("*.ipynb")}
        missing_nbs = sorted(expected_notebooks - actual_notebooks)
        extra_nbs = sorted(actual_notebooks - expected_notebooks)
        for nb in missing_nbs:
            failures.append(f"missing submodule-aligned tutorial notebook: tutorials/{module_name}/{nb}")
        for nb in extra_nbs:
            failures.append(f"non-aligned tutorial notebook name: tutorials/{module_name}/{nb}")

        overview_path = tutorials / f"Document_{mod_title}_Overview.ipynb"
        if overview_path.is_file():
            ov_text = overview_path.read_text(encoding="utf-8")
            for required_section in ("## Choose a guide", "## Remember"):
                if required_section not in ov_text:
                    failures.append(
                        f"overview notebook tutorials/{module_name}/{overview_path.name} is missing section '{required_section}'"
                    )

        for stem, sub_exports in submodules.items():
            nb_name = f"Document_{mod_title}_{_title_stem(stem)}.ipynb"
            nb_path = tutorials / nb_name
            if nb_path.is_file():
                sub_text = nb_path.read_text(encoding="utf-8")
                for api_name in sub_exports:
                    if not re.search(rf"\b{re.escape(api_name)}\b", sub_text):
                        failures.append(
                            f"submodule tutorial tutorials/{module_name}/{nb_name} is missing API '{api_name}' from klygo/{module_name}/{stem}.py"
                        )
                for required_section in ("## Jump to an API", "## Put it together", "## Remember"):
                    if required_section not in sub_text:
                        failures.append(
                            f"submodule tutorial tutorials/{module_name}/{nb_name} is missing section '{required_section}'"
                        )

        # Check that each tutorial notebook contains its expected Colab URL
        for nb_path in tutorials.glob("*.ipynb"):
            expected_url = f"{COLAB_BASE}/{module_name}/{nb_path.name}"
            nb_content = nb_path.read_text(encoding="utf-8")
            if expected_url not in nb_content:
                failures.append(
                    f"tutorial notebook missing Colab URL '{expected_url}': tutorials/{module_name}/{nb_path.name}"
                )

    # 3. Markdown local links and code fence balance in docs/<module>/
    if docs_dir.exists():
        for doc_path in docs_dir.rglob("*.md"):
            text = doc_path.read_text(encoding="utf-8", errors="ignore")
            fences = re.findall(r"^```[^`\r\n]*$", text, flags=re.MULTILINE)
            if len(fences) % 2 != 0:
                failures.append(f"unclosed code fence in docs: {doc_path.relative_to(root)}")
            for target in re.findall(r"\[[^\]]+\]\(([^)\s]+)\)", text):
                parsed = urlsplit(target)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                resolved = (doc_path.parent / unquote(parsed.path)).resolve()
                if not resolved.exists():
                    failures.append(f"broken local link in {doc_path.relative_to(root)}: {target}")

    # 4. English language standard for all raise error statements in implementation
    if module_dir.exists():
        for py_file in module_dir.rglob("*.py"):
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Raise):
                        for child in ast.walk(node):
                            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                                if _contains_non_english(child.value):
                                    failures.append(
                                        f"raise error message must be in English: {py_file.relative_to(root)}:{node.lineno}"
                                    )
            except Exception:
                pass

    return len(exports), len(submodules), failures


class TestPublicModuleDocumentation(unittest.TestCase):
    """Unittest test suite to verify documentation contracts across all modules."""

    def test_all_public_modules(self):
        root = Path(__file__).resolve().parents[4] if "antigravity" not in str(Path(__file__)) else Path(r"c:\Users\Mine\Downloads\klygo")
        for mod in ALL_MODULES:
            with self.subTest(module=mod):
                exports_count, sub_count, failures = audit_module(mod, root, run_examples=False)
                self.assertEqual(failures, [], f"Module '{mod}' failed audit:\n" + "\n".join(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Klygo module documentation and tutorial standards.")
    parser.add_argument("module", nargs="?", default=None, help="Klygo module name (e.g. files, archive, config)")
    parser.add_argument("--all", action="store_true", help="Audit all public modules in Klygo")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--skip-examples", action="store_true", help="Skip executing example files")
    parser.add_argument("--sync-colab", action="store_true", help="Synchronize and fix Colab badges and links across all notebooks")
    args = parser.parse_args()

    root = args.root.resolve()
    tut_dir = root / "tutorials"

    if args.sync_colab:
        synced_count = 0
        if tut_dir.exists():
            for nb_path in tut_dir.rglob("*.ipynb"):
                if sync_colab_in_notebook(nb_path, tut_dir):
                    synced_count += 1
        print(f"Synchronized Colab badges and links across {synced_count} notebook(s).\n")

    target_modules = ALL_MODULES if args.all or args.module in (None, "all") else [args.module]

    total_failures = 0
    print(f"Auditing {len(target_modules)} Klygo module(s)...\n")

    for mod in target_modules:
        print(f"--- klygo.{mod} ---")
        exports_count, sub_count, failures = audit_module(mod, root, run_examples=not args.skip_examples)
        tut_mod_dir = root / "tutorials" / mod
        tut_count = len(list(tut_mod_dir.glob("*.ipynb"))) if tut_mod_dir.exists() else 0
        print(f"klygo.{mod}: {exports_count} public APIs across {sub_count} submodules, {tut_count} tutorial notebooks")
        if not tut_mod_dir.exists():
            print(f"NOTICE: tutorials/{mod} does not exist yet")
        if failures:
            total_failures += len(failures)
            for failure in failures:
                print(f"ERROR: {failure}")
        else:
            print("PASS: source, docs, examples, dogfooding, and tutorial notebooks verified.")
        print()

    if total_failures:
        print(f"FAILED: {total_failures} error(s) detected across audited modules.")
        return 1

    print("ALL MODULES PASSED: All documentation, examples, and tutorial contracts are satisfied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
