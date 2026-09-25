# `files.parent`

```python
files.parent(path)
```

Return the immediate parent Path.

## Contract

This is a pure path operation and does not require the path to exist.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `path`: `str | pathlib.Path` — filesystem path to inspect or operate on.

## Returns

A `pathlib.Path` representing the result.

## Errors and edge cases

This helper normally does not access the filesystem. Invalid argument types may raise `TypeError`.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.parent(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`parent.py`](../../../examples/files/parent.py) for an executable example.

## Tests

See [`test_parent.py`](../../../test/files/test_parent.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.parent."""

from klygo import files

print(files.parent("dataset/images/sample.jpg"))


# Additional cases
assert files.parent("dataset/images/cat.jpg").name == "images"
assert files.parent("cat.jpg") == files.path(".")
```
