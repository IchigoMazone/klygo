---
name: klygo-module-guides
description: Complete or upgrade a public Klygo module with consistent API code comments and docstrings, unittest coverage, runnable examples, English API documentation, and code-heavy tutorial notebooks. Use when finishing one Klygo module or applying the same documentation and validation standard to a new module.
---

# Klygo Module Guides

Build one module as a coherent public product. Treat its source, tests, examples,
reference docs, and tutorials as five views of the same API. Derive every claim
and example from the checked-out implementation and `__all__`.

Read [references/authoring-standard.md](references/authoring-standard.md) before
editing a module. Run `scripts/audit_public_api.py` after the module artifacts
exist.

## Workflow

1. Inspect repository instructions, `klygo/<module>/__init__.py`, every exported
   implementation file (`klygo/<module>/<submodule>.py`), dependencies, existing
   tests, examples, docs, and tutorials. Record the exact public inventory from
   `__all__` and map each exported symbol to its public implementation submodule.
2. Fix the implementation and public documentation contract together. Keep API
   names, signatures, exceptions, return shapes, and side effects aligned:
   - Group the public inventory in `klygo/<module>/__init__.py` by its implementation
     submodules (`io.py`, `metadata.py`, `operations.py`, `context.py`, etc.) and
     define `__all__` in each public submodule.
   - **Cross-module & intra-module reuse (dogfooding)**: Every module and submodule
     must use Klygo's own public capabilities (`klygo.files`, `klygo.utils`,
     `klygo.validators`, sibling submodules, etc.) wherever applicable rather than
     duplicating logic or calling raw `pathlib.Path`, `os.path`, or `shutil` directly
     at the orchestration layer.
   - Add code comments only where an invariant, safety decision, lazy behavior, or
     non-obvious compatibility rule needs explanation.
   - **Language consistency (English-only standard)**:
     - All exception and error messages (`raise ValueError(...)`, `raise TypeError(...)`,
       `raise KeyError(...)`, etc.) across the entire codebase must be authored in clear,
       concise, professional English. No non-English or mixed-language error strings are permitted.
     - All public API docstrings, parameter definitions, return descriptions, and official
       API comments must strictly be written in English. Internal comments explaining
       invariants or safety logic must also be in English.
3. Write or update one meaningful `unittest` suite. Cover observable behavior,
   supported `str` and `os.PathLike[str]` inputs, overwrite/destructive rules,
   lazy iteration, optional dependencies, and representative failures as relevant.
   Do not add pytest or duplicate implementation branches mechanically.
4. Create or update a runnable example and reference page for every public API,
   plus the module indexes (`docs/<module>/README.md`), using the exact same
   submodule grouping as `klygo/<module>/__init__.py`. Examples must use public
   imports such as `from klygo import archive, files` and dogfood Klygo's own APIs
   (`files.path()`, `files.save()`, `files.load()`, `files.mkdir()`, etc.) instead
   of importing `pathlib.Path` or calling raw `Path` I/O methods directly.
   **Pure documentation boundary**: The `docs/` tree must strictly contain `.md`
   markdown files and documentation directories. Never add test scripts (such as
   `test_documentation.py`) or Python runtime artifacts (`__pycache__`) into `docs/`.
   All documentation contracts, docstring conformance, and link integrity are tested
   centrally via `scripts/audit_public_api.py`.
5. Run the module tests and documentation checks using the centralized skill script:
   `python .agents/skills/klygo-module-guides/scripts/audit_public_api.py <module>`
   (or `--all` for repository-wide verification). Fix all failures before creating
   tutorial notebooks. This is a hard gate: a tutorial must not document an
   unverified implementation.
6. Create code-heavy notebooks under `tutorials/<module>/` whose filenames mirror
   the module's public implementation files 1-to-1:
   - **Colab GitHub integration standard**: All Google Colab URLs must point directly
     to the official GitHub repository path on branch `v2-release`:
     `https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/<module>/<Notebook>.ipynb`
     Include the standard Colab badge at the top of markdown cells:
     `[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/<module>/<Notebook>.ipynb)`
     This eliminates manual Google Drive uploads and sync scripts; any git push
     to the repository automatically deploys and synchronizes live, executable
     notebooks on Google Colab.
     **Automated Colab synchronization**: Use `scripts/audit_public_api.py --sync-colab`
     to automatically check, inject, and fix Colab badges and cross-navigation links
     across all notebooks repository-wide.
   - Overview notebook: `Document_<Module>_Overview.ipynb`
     - Include a `## Choose a guide` table (`| Submodule | Notebook | Colab | Use it for | APIs |`)
       linking to every `Document_<Module>_<Submodule>.ipynb` and its Colab link.
     - End with a `## Remember` section summarizing module conventions and linking to
       sibling Klygo module overviews (`Document_Files_Overview.ipynb`,
       `Document_Archive_Overview.ipynb`, `Document_Config_Overview.ipynb`, etc.) so
       readers can switch flexibly across modules.
   - Submodule notebooks: `Document_<Module>_<Submodule>.ipynb`, where `<Submodule>`
     is the Title_Case stem of `klygo/<module>/<submodule>.py` (for example,
     `klygo/archive/io.py` -> `Document_Archive_Io.ipynb`,
     `klygo/config/access.py` -> `Document_Config_Access.ipynb`).
     - Begin with a single install cell for the selected Git revision, followed by a
       resettable temporary workspace built with `files.path()` and a `## Jump to an API`
       navigation bar.
     - Give every API in that submodule a discoverable `### <module>.<api>` section with
       verified signature, parameter bullets, return summary, basic use, practical
       variation, and caught examples of specific user-facing errors.
     - End every submodule notebook with `## Put it together` (a combined workflow code
       cell) and `## Remember` (key behavioral takeaways plus navigation links to the
       next/related `Document_<Module>_<Submodule>.ipynb`, `Document_<Module>_Overview.ipynb`,
       and related Klygo modules for seamless cross-module navigation).
7. Execute every notebook top to bottom twice from clean notebook state. Validate
   API coverage, 1-to-1 submodule filename alignment, required sections (`## Choose a guide`,
   `## Jump to an API`, `## Put it together`, `## Remember`), cross-module dogfooding,
   notebook structure, English-only authored prose, safe paths, and rerun behavior.
   State local versus Colab validation honestly.
8. Run the relevant module suite, repository-wide public API and documentation checks
   (`python .agents/skills/klygo-module-guides/scripts/audit_public_api.py --all`),
   and `git diff --check` once more. Summarize files changed, API count, test
   count, notebook execution, and any optional backend behavior not exercised.

Do not push, publish, or alter unrelated modules unless the user requests it.
