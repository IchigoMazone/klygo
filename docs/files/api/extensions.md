# `files.extensions`

```python
files.extensions(value)
```

Return all filename suffixes as a tuple.

## Contract

For archive.tar.gz, returns ('.tar', '.gz').

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `value`: `str | pathlib.Path` — path-like value to transform.

## Returns

A tuple containing every suffix.

## Errors and edge cases

This helper normally does not access the filesystem. Invalid argument types may raise `TypeError`.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.extensions(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`extensions.py`](../../../examples/files/extensions.py) for an executable example.

## Tests

See [`test_extensions.py`](../../../test/files/test_extensions.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.extensions."""

from klygo import files

print(files.extensions("archives/dataset.tar.gz"))


# Additional cases
assert files.extensions("photo.jpg") == (".jpg",)
assert files.extensions("README") == ()
```
