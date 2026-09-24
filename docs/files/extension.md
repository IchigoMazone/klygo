# `files.extension`

```python
files.extension(path)
```

Return the final filename extension.

## Contract

For archive.tar.gz, the result is .gz.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `path`: `str | pathlib.Path` — filesystem path to inspect or operate on.

## Returns

A string result.

## Errors and edge cases

This helper normally does not access the filesystem. Invalid argument types may raise `TypeError`.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.extension(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`extension.py`](../../examples/files/extension.py) for an executable example.

## Tests

See [`test_extension.py`](../../test/files/test_extension.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.extension."""

from klygo import files

print(files.extension("dataset/archive.tar.gz"))


# Additional cases
assert files.extension("photo.jpg") == ".jpg"
assert files.extension("README") == ""
```
