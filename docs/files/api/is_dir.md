# `files.is_dir`

```python
files.is_dir(path)
```

Check whether a path points to a directory.

## Contract

Returns False for missing paths and regular files.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `path`: `str | pathlib.Path` — filesystem path to inspect or operate on.

## Returns

A boolean result.

## Errors and edge cases

This helper normally does not access the filesystem. Invalid argument types may raise `TypeError`.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.is_dir(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`is_dir.py`](../../../examples/files/is_dir.py) for an executable example.

## Tests

See [`test_is_dir.py`](../../../test/files/test_is_dir.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.is_dir."""

from klygo import files

print(files.is_dir("klygo"))


# Additional cases
assert not files.is_dir("README.md")
assert not files.is_dir("missing-directory")
```
