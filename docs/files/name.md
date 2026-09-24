# `files.name`

```python
files.name(path)
```

Return the final path component including its extension.

## Contract

This is a pure path operation and does not require the path to exist.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `path`: `str | pathlib.Path` — filesystem path to inspect or operate on.

## Returns

A string result.

## Errors and edge cases

This helper normally does not access the filesystem. Invalid argument types may raise `TypeError`.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.name(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`name.py`](../../examples/files/name.py) for an executable example.

## Tests

See [`test_name.py`](../../test/files/test_name.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.name."""

from klygo import files

print(files.name("dataset/images/sample.jpg"))


# Additional cases
assert files.name("archive.tar.gz") == "archive.tar.gz"
assert files.name("README") == "README"
```
