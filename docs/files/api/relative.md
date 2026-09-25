# `files.relative`

```python
files.relative(value, start='.')
```

Compute a path relative to a starting directory.

## Contract

Uses filesystem path semantics but does not require either path to exist.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `value`: `str | pathlib.Path` — path-like value to transform.
- `start`: `str | int` — starting directory for `relative`, or initial suffix number for `unique_path`.

## Returns

A `pathlib.Path` representing the result.

## Errors and edge cases

This helper normally does not access the filesystem. Invalid argument types may raise `TypeError`.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.relative(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`relative.py`](../../../examples/files/relative.py) for an executable example.

## Tests

See [`test_relative.py`](../../../test/files/test_relative.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.relative."""

from klygo import files

print(files.relative("dataset/images/train", "dataset"))


# Additional cases
assert files.relative("dataset/labels/train", "dataset") == files.path("labels/train")
assert files.relative("dataset", "dataset") == files.path(".")
```
