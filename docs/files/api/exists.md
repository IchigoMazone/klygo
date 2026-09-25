# `files.exists`

```python
files.exists(path)
```

Check whether a filesystem path exists.

## Contract

Returns a boolean and does not raise when the path is absent.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `path`: `str | pathlib.Path` — filesystem path to inspect or operate on.

## Returns

A boolean result.

## Errors and edge cases

This helper normally does not access the filesystem. Invalid argument types may raise `TypeError`.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.exists(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`exists.py`](../../../examples/files/exists.py) for an executable example.

## Tests

See [`test_exists.py`](../../../test/files/test_exists.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.exists."""

from klygo import files

print(files.exists("README.md"))
print(files.exists("missing-file"))


# Additional cases
from pathlib import Path

assert files.exists(Path("README.md"))
assert not files.exists("path-that-does-not-exist")
```
