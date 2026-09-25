# `files.normalize`

```python
files.normalize(value)
```

Normalize separators and dot components.

## Contract

Does not force the result to be absolute or require it to exist.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `value`: `str | pathlib.Path` — path-like value to transform.

## Returns

A `pathlib.Path` representing the result.

## Errors and edge cases

This helper normally does not access the filesystem. Invalid argument types may raise `TypeError`.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.normalize(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`normalize.py`](../../../examples/files/normalize.py) for an executable example.

## Tests

See [`test_normalize.py`](../../../test/files/test_normalize.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.normalize."""

from klygo import files

print(files.normalize("dataset/images/../labels"))


# Additional cases
assert files.normalize("./dataset/./images") == files.path("dataset/images")
assert files.normalize("dataset/images/../labels") == files.path("dataset/labels")
```
