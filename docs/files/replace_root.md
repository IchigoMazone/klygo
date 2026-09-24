# `files.replace_root`

```python
files.replace_root(value, old_root, new_root)
```

Map a path from one directory tree into another.

## Contract

Preserves the relative location and raises ValueError when value is outside old_root.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `value`: `str | pathlib.Path` — path-like value to transform.
- `old_root`: `str | pathlib.Path` — root removed from the source path.
- `new_root`: `str | pathlib.Path` — replacement root.

## Returns

A `pathlib.Path` representing the result.

## Errors and edge cases

Raises `ValueError` when the input is outside `old_root`.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.replace_root(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`replace_root.py`](../../examples/files/replace_root.py) for an executable example.

## Tests

See [`test_replace_root.py`](../../test/files/test_replace_root.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.replace_root."""

from klygo import files

label = files.replace_root(
    "dataset/images/train/cat.jpg",
    "dataset/images",
    "dataset/labels",
)
print(label)


# Additional cases
assert label == files.path("dataset/labels/train/cat.jpg")
try:
    files.replace_root("outside/cat.jpg", "dataset/images", "dataset/labels")
except ValueError:
    print("source must be inside old_root")
```
