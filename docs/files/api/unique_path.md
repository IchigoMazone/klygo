# `files.unique_path`

```python
files.unique_path(value, separator='_', start=1)
```

Return a currently unused path.

## Contract

Adds an incrementing numeric suffix while preserving known compound extensions. Broken symlinks occupy names too. Checks the filesystem, but does not create or reserve the returned path; another writer can claim it afterward.

Inputs accept strings and text-returning `os.PathLike` objects (including `pathlib.Path`), using the current operating system's path rules. Byte paths are not supported.

## Parameters

- `value`: `str | os.PathLike[str]` — path-like value to transform.
- `separator`: `str` — separator before the numeric suffix.
- `start`: `int`, default `1` — initial numeric suffix; must be non-negative.

## Returns

A `pathlib.Path` representing the result.

## Errors and edge cases

Raises `ValueError` when `start` is negative.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.unique_path(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`unique_path.py`](../../../examples/files/unique_path.py) for an executable example.

## Tests

See [`test_unique_path.py`](../../../test/files/test_unique_path.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.unique_path."""

from tempfile import TemporaryDirectory
from klygo import files

with TemporaryDirectory() as directory:
    existing = files.path(directory) / "result.json"
    files.save(existing, {"ok": True}, verbose=False)
    print(files.unique_path(existing))


# Additional cases
# Compound extensions remain intact when numbering is needed.
with TemporaryDirectory() as directory:
    existing = files.path(directory) / "dataset.tar.gz"
    existing.touch()
    assert files.name(files.unique_path(existing)) == "dataset_1.tar.gz"
    assert files.name(files.unique_path(existing, separator="-", start=4)) == "dataset-4.tar.gz"
```
