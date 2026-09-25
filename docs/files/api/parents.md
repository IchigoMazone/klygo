# `files.parents`

```python
files.parents(value)
```

Return every ancestor Path from nearest to farthest.

## Contract

The result is an immutable tuple.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `value`: `str | pathlib.Path` — path-like value to transform.

## Returns

A tuple of ancestor paths.

## Errors and edge cases

This helper normally does not access the filesystem. Invalid argument types may raise `TypeError`.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.parents(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`parents.py`](../../../examples/files/parents.py) for an executable example.

## Tests

See [`test_parents.py`](../../../test/files/test_parents.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.parents."""

from klygo import files

for ancestor in files.parents("dataset/images/train/cat.jpg"):
    print(ancestor)


# Additional cases
ancestors = files.parents("dataset/images/train/cat.jpg")
assert ancestors[0] == files.path("dataset/images/train")
assert ancestors[1] == files.path("dataset/images")
```
