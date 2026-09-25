# `files.remove`

```python
files.remove(path, recursive=True, missing_ok=True)
```

Remove a file, symlink, or directory.

## Contract

Directories are removed recursively by default. Missing paths may be ignored.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `path`: `str | pathlib.Path` — filesystem path to inspect or operate on.
- `recursive`: `bool` — include descendants or recursively remove a directory.
- `missing_ok`: `bool` — tolerate an absent target.

## Returns

`None`; the filesystem is modified as a side effect.

## Errors and edge cases

Raises `FileNotFoundError` only when the target is absent and `missing_ok=False`. Non-empty directories fail when `recursive=False`.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.remove(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`remove.py`](../../../examples/files/remove.py) for an executable example.

## Tests

See [`test_remove.py`](../../../test/files/test_remove.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.remove."""

from tempfile import TemporaryDirectory
from pathlib import Path
from klygo import files

with TemporaryDirectory() as directory:
    target = Path(directory) / "temporary"
    target.mkdir()
    files.remove(target)
    print(target.exists())


# Additional cases
# missing_ok controls whether an absent path is an error.
with TemporaryDirectory() as directory:
    missing = Path(directory) / "missing"
    files.remove(missing)
    try:
        files.remove(missing, missing_ok=False)
    except FileNotFoundError:
        print("missing path reported")
```
