"""Audit public API coverage, 1-to-1 submodule alignment, and cross-module dogfooding across Klygo."""

from __future__ import annotations

import argparse
import importlib
import inspect
import re
import sys
from pathlib import Path


def _title_stem(stem: str) -> str:
    return "_".join(part.capitalize() for part in stem.split("_"))


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("module", help="Klygo module name, for example archive")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.root.resolve()
    sys.path.insert(0, str(root))
    public_module = importlib.import_module(f"klygo.{args.module}")
    exports = list(getattr(public_module, "__all__", ()))
    if not exports:
        print(f"ERROR: klygo.{args.module} has no public __all__ inventory")
        return 1

    module_dir = root / "klygo" / args.module
    docs = root / "docs" / args.module / "api"
    examples = root / "examples" / args.module
    tutorials = root / "tutorials" / args.module
    notebook_text = ""
    for notebook in sorted(tutorials.glob("*.ipynb")) if tutorials.exists() else ():
        notebook_text += notebook.read_text(encoding="utf-8")

    failures: list[str] = []
    for name in exports:
        value = getattr(public_module, name, None)
        if value is None:
            failures.append(f"missing exported attribute: {name}")
            continue
        if not inspect.getdoc(value):
            failures.append(f"missing public docstring: {name}")
        if not (docs / f"{name}.md").is_file():
            failures.append(f"missing API doc: docs/{args.module}/api/{name}.md")
        example_file = examples / f"{name}.py"
        if not example_file.is_file():
            failures.append(f"missing example: examples/{args.module}/{name}.py")
        else:
            ex_text = example_file.read_text(encoding="utf-8")
            if re.search(r"^\s*from\s+pathlib\s+import\s+Path\b", ex_text, flags=re.MULTILINE):
                failures.append(
                    f"example imports pathlib.Path directly instead of using klygo.files.path(): examples/{args.module}/{name}.py"
                )
        if notebook_text and not re.search(rf"\b{re.escape(name)}\b", notebook_text):
            failures.append(f"missing tutorial mention: {name}")

    # Verify 1-to-1 submodule <-> tutorial notebook naming and API coverage
    submodules = _discover_public_submodules(args.module, module_dir, set(exports))
    mod_title = _title_stem(args.module)
    if tutorials.exists() and submodules:
        expected_notebooks = {f"Document_{mod_title}_Overview.ipynb"} | {
            f"Document_{mod_title}_{_title_stem(stem)}.ipynb" for stem in submodules
        }
        actual_notebooks = {p.name for p in tutorials.glob("*.ipynb")}
        missing_nbs = sorted(expected_notebooks - actual_notebooks)
        extra_nbs = sorted(actual_notebooks - expected_notebooks)
        for nb in missing_nbs:
            failures.append(f"missing submodule-aligned tutorial notebook: tutorials/{args.module}/{nb}")
        for nb in extra_nbs:
            failures.append(f"non-aligned tutorial notebook name: tutorials/{args.module}/{nb}")

        overview_path = tutorials / f"Document_{mod_title}_Overview.ipynb"
        if overview_path.is_file():
            ov_text = overview_path.read_text(encoding="utf-8")
            for required_section in ("## Choose a guide", "## Remember"):
                if required_section not in ov_text:
                    failures.append(
                        f"overview notebook tutorials/{args.module}/{overview_path.name} is missing section '{required_section}'"
                    )

        for stem, sub_exports in submodules.items():
            nb_name = f"Document_{mod_title}_{_title_stem(stem)}.ipynb"
            nb_path = tutorials / nb_name
            if nb_path.is_file():
                sub_text = nb_path.read_text(encoding="utf-8")
                for api_name in sub_exports:
                    if not re.search(rf"\b{re.escape(api_name)}\b", sub_text):
                        failures.append(
                            f"submodule tutorial tutorials/{args.module}/{nb_name} is missing API '{api_name}' from klygo/{args.module}/{stem}.py"
                        )
                for required_section in ("## Jump to an API", "## Put it together", "## Remember"):
                    if required_section not in sub_text:
                        failures.append(
                            f"submodule tutorial tutorials/{args.module}/{nb_name} is missing section '{required_section}'"
                        )

    print(f"klygo.{args.module}: {len(exports)} public APIs across {len(submodules)} submodules ({', '.join(submodules)})")
    if not tutorials.exists():
        print(f"NOTICE: tutorials/{args.module} does not exist yet")
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1
    print("PASS: source, docs, examples, cross-module dogfooding, and submodule-aligned tutorials cover __all__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
