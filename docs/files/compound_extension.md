# `files.compound_extension`

```python
files.compound_extension(value)
```

Return all filename suffixes joined together.

## Contract

For archive.tar.gz, returns .tar.gz.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `value`: `str | pathlib.Path` — path-like value to transform.

## Returns

A string result.

## Errors and edge cases

This helper normally does not access the filesystem. Invalid argument types may raise `TypeError`.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.compound_extension(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`compound_extension.py`](../../examples/files/compound_extension.py) for an executable example.

## Tests

See [`test_compound_extension.py`](../../test/files/test_compound_extension.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.compound_extension."""

from klygo import files

print(files.compound_extension("archives/dataset.tar.gz"))


# Additional cases
assert files.compound_extension("photo.jpg") == ".jpg"
assert files.compound_extension("README") == ""
```
