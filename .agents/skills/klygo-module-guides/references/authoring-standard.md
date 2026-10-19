# Klygo module authoring standard

## Source, cross-module reuse, and public API

- `klygo/<module>/__init__.py` and its `__all__` define the supported inventory.
- Every public implementation submodule `klygo/<module>/<submodule>.py` defines its
  own `__all__` listing the symbols it contributes to `klygo.<module>.__all__`.
- Public code accepts the path-like types promised by its annotations and docs.
- **Cross-module & intra-module dogfooding**:
  - Modules must build upon other Klygo modules and sibling submodules whenever
    possible: use `klygo.files` (`files.path`, `files.exists`, `files.is_file`,
    `files.is_dir`, `files.is_symlink`, `files.mkdir`, `files.remove`, `files.copy`,
    `files.move`, `files.rename`, `files.find`, `files.load`, `files.save`,
    `files.name`, `files.stem`, `files.extension`, `files.compound_extension`,
    `files.parent`, `files.resolve`, `files.is_within`, etc.), `klygo.utils`
    (`human_size`, `ProgressBar`), and `klygo.validators` (`validate_type`) instead
    of re-implementing them with raw `pathlib.Path`, `os.path`, or `shutil` calls.
  - Within `klygo/files/` itself, higher-level submodules (`io.py`, `metadata.py`,
    `download.py`, `filesystem.py`) reuse the path and filesystem primitives defined
    in `paths.py` and `filesystem.py`.
  - Within user-facing examples and tutorials, import through `from klygo import <module>, files`
    and construct/manipulate paths and sample files via `files.path()`, `files.mkdir()`,
    `files.save()`, `files.load()`, etc., instead of importing `pathlib.Path` directly
    (except in guides specifically dedicated to `pathlib.Path` interoperability).
- Do not retain compatibility aliases when the module intentionally renamed an API.
- Comments explain why a non-obvious block exists. Do not narrate obvious syntax.

## Public docstrings

Use concise English NumPy-style docstrings. Include applicable sections in this
order; omit sections that genuinely do not apply:

```text
"""State what the API returns or changes.

Parameters
----------
source : str or os.PathLike[str]
    Explain accepted input and practical meaning.

Returns
-------
pathlib.Path
    Explain the returned value, including whether it was created.

Raises
------
FileNotFoundError
    Explain the exact missing input.
ValueError
    Explain invalid values or unsupported states.

Notes
-----
Document safety, lazy behavior, optional dependencies, or platform behavior.

Examples
--------
>>> from klygo import module
>>> result = module.operation("input")
"""
```

Signatures, defaults, supported formats, return types, and exceptions must match
source behavior. Do not promise stronger atomicity, security, memory use, or
cross-platform behavior than the implementation provides.

## Tests

- Use `unittest` consistently with the repository.
- Create data in `tempfile.TemporaryDirectory`; never modify user paths.
- Test public calls rather than private helpers unless a backend contract requires it.
- Include successful use, important options, returned values, side effects, and
  representative specific exceptions.
- Test optional backends conditionally and report skipped capabilities.
- Run the focused module suite first, then the repository suite when changes can
  affect shared APIs.

## Examples

- Maintain one executable file per exported API under `examples/<module>/`.
- Examples should show realistic work, output inspection, and one useful option or
  recovery path when the API is complex.
- Dogfood Klygo public APIs (`files.path`, `files.save`, `files.load`, `files.mkdir`,
  etc.) instead of `from pathlib import Path` or `.write_text()` / `.read_text()`.
- Use descriptive variables and specific exception handling. Avoid assertion-heavy
  examples that read like tests.
- Keep helper files private with a leading underscore and exclude them from the
  public inventory.

## Reference documentation

- Maintain `docs/<module>/README.md` as the task-oriented index.
- Maintain `docs/<module>/api/<name>.md` for each export.
- Group public APIs in `klygo/<module>/__init__.py` and `docs/<module>/README.md`
  by their public implementation submodules (`klygo/<module>/<submodule>.py`) so
  source files, reference docs, and tutorial notebooks share an identical mental map.
- Each page gives the verified signature, purpose, parameters, return/side effects,
  relevant errors, at least one runnable example, and related APIs.
- Link examples and tutorials from the module README. Use English throughout.

## Tutorials

- Create one overview notebook named `Document_<Module>_Overview.ipynb` and one
  focused notebook per public implementation submodule in `klygo/<module>/`.
- Follow the strict 1-to-1 naming rule `Document_<Module>_<Submodule>.ipynb`, where
  `<Module>` and `<Submodule>` are Title_Case underscore-separated names matching
  `klygo/<module>/<submodule>.py` (excluding `__init__.py` and private `_*` modules):
  - `klygo/files/{paths,filesystem,metadata,io,download}.py` ->
    `Document_Files_{Overview,Paths,Filesystem,Metadata,Io,Download}.ipynb`
  - `klygo/archive/{io,metadata,operations,context}.py` ->
    `Document_Archive_{Overview,Io,Metadata,Operations,Context}.ipynb`
  - `klygo/config/{io,creation,mapping,access,inspection,structure,environment,validation,config}.py` ->
    `Document_Config_{Overview,Io,Creation,Mapping,Access,Inspection,Structure,Environment,Validation,Config}.ipynb`
- Every public API exported by `klygo/<module>/<submodule>.py` must be documented
  in the corresponding `Document_<Module>_<Submodule>.ipynb` notebook.
- **Required sections in `Document_<Module>_Overview.ipynb`**:
  - `## Choose a guide`: Markdown table (`| Submodule | Notebook | Colab | Use it for | APIs |`)
    linking every `Document_<Module>_<Submodule>.ipynb` and its Colab URL.
  - `## Try the public API`: Quick runnable demo using `files.path(tutorial_workspace.name)`.
  - `## Remember`: Summary of module conventions plus direct links to the overview
    notebooks of sibling Klygo modules (`klygo.files`, `klygo.archive`, `klygo.config`, etc.)
    for flexible cross-module switching.
- **Required sections in every `Document_<Module>_<Submodule>.ipynb`**:
  - Install cell (`!pip install --upgrade --no-cache-dir git+https://github.com/IchigoMazone/klygo.git@<revision>`).
  - Setup cell (`from klygo import <module>, files` and `root = files.path(tutorial_workspace.name)`).
  - `## Jump to an API`: Anchor links to each API in the notebook.
  - `### <module>.<api>`: Verified signature (using clean public types like `pathlib.Path`),
    parameter bullets, return summary, **Basic use**, **Practical variation**, and
    **Expected error and recovery** cells.
  - `## Put it together`: A combined code cell demonstrating the submodule's APIs working together.
  - `## Remember`: Key behavioral takeaways for the submodule, links to the next/related
    submodule notebook and `Document_<Module>_Overview.ipynb` (plus Colab link when available),
    and cross-module links to related Klygo modules (`klygo.files`, `klygo.archive`,
    `klygo.config`) so users can navigate seamlessly across both submodules and modules.
- Never use live private data, credentials, Drive mounts, or arbitrary deletion.
- Network examples should have a local deterministic equivalent; keep live URL code
  clearly marked when it is intentionally not executed.
- Run all cells twice. Do not claim Colab validation after local-only execution.

## Completion gate

A module is complete when its `__all__` inventory matches source, examples, API
pages, tests, and tutorials; tutorial notebook filenames match
`Document_<Module>_Overview.ipynb` plus every public `klygo/<module>/<submodule>.py`;
every notebook includes its required `## Choose a guide` / `## Jump to an API` /
`## Put it together` / `## Remember` sections; cross-module dogfooding is verified;
focused tests pass; notebooks execute twice; no temporary generator scripts remain;
and Git shows only intended artifacts.
