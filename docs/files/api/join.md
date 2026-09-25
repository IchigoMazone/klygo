# `files.join`

```python
files.join(*parts)
```

Join one or more path components.

## Contract

At least one component is required. The result follows platform path rules.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `parts`: `*str | *pathlib.Path` — one or more components.

## Returns

A `pathlib.Path` representing the result.

## Errors and edge cases

Raises `ValueError` when no components are supplied.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.join(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`join.py`](../../../examples/files/join.py) for an executable example.

## Tests

See [`test_join.py`](../../../test/files/test_join.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.join."""

from klygo import files

print(files.join("dataset", "images", "train", "sample.jpg"))


# Additional cases
assert files.join("dataset", "labels") == files.path("dataset/labels")
try:
    files.join()
except ValueError:
    print("at least one component is required")
```
