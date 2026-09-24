# `files.unique_path`

```python
files.unique_path(value, separator='_', start=1)
```

Return a currently unused path.

## Contract

Adds an incrementing numeric suffix while preserving known compound extensions.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `value`: `str | pathlib.Path` — path-like value to transform.
- `separator`: `str` — separator before the numeric suffix.
- `start`: `str | int` — starting directory for `relative`, or initial suffix number for `unique_path`.

## Returns

A `pathlib.Path` representing the result.

## Errors and edge cases

Raises `ValueError` when `start` is negative.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.unique_path(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`unique_path.py`](../../examples/files/unique_path.py) for an executable example.

## Tests

See [`test_unique_path.py`](../../test/files/test_unique_path.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.unique_path."""

from tempfile import TemporaryDirectory
from pathlib import Path
from klygo import files

with TemporaryDirectory() as directory:
    existing = Path(directory) / "result.json"
    existing.touch()
    print(files.unique_path(existing))


# Additional cases
# Compound extensions remain intact when numbering is needed.
with TemporaryDirectory() as directory:
    existing = Path(directory) / "dataset.tar.gz"
    existing.touch()
    assert files.unique_path(existing).name == "dataset_1.tar.gz"
    assert files.unique_path(existing, separator="-", start=4).name == "dataset-4.tar.gz"
```
