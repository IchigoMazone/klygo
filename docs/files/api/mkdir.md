# `files.mkdir`

```python
files.mkdir(path, parents=True, exist_ok=True)
```

Create a directory and return its Path.

## Contract

Can create missing parents and optionally tolerate an existing directory.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `path`: `str | pathlib.Path` — filesystem path to inspect or operate on.
- `parents`: `bool` — create missing parent directories.
- `exist_ok`: `bool` — tolerate an existing directory.

## Returns

A `pathlib.Path` representing the result.

## Errors and edge cases

May raise `FileExistsError`, `FileNotFoundError`, or `PermissionError` according to `parents` and `exist_ok`.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.mkdir(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`mkdir.py`](../../../examples/files/mkdir.py) for an executable example.

## Tests

See [`test_mkdir.py`](../../../test/files/test_mkdir.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.mkdir."""

from tempfile import TemporaryDirectory
from pathlib import Path
from klygo import files

with TemporaryDirectory() as directory:
    created = files.mkdir(Path(directory) / "outputs" / "images")
    print(created, created.is_dir())


# Additional cases
# Reusing the same path is safe with the default exist_ok=True.
with TemporaryDirectory() as directory:
    target = Path(directory) / "cache"
    assert files.mkdir(target) == target
    assert files.mkdir(target) == target
```
