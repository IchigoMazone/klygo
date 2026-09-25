# `files.stem`

```python
files.stem(path)
```

Return the final path component without its last extension.

## Contract

For archive.tar.gz, the result is archive.tar.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `path`: `str | pathlib.Path` — filesystem path to inspect or operate on.

## Returns

A string result.

## Errors and edge cases

This helper normally does not access the filesystem. Invalid argument types may raise `TypeError`.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.stem(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`stem.py`](../../../examples/files/stem.py) for an executable example.

## Tests

See [`test_stem.py`](../../../test/files/test_stem.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.stem."""

from klygo import files

print(files.stem("dataset/archive.tar.gz"))


# Additional cases
assert files.stem("photo.jpg") == "photo"
assert files.stem("README") == "README"
```
