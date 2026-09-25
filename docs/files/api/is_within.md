# `files.is_within`

```python
files.is_within(value, root, resolve_paths=True)
```

Check whether a path is contained by a root directory.

## Contract

Resolution is enabled by default to handle dot components and symlinks safely.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `value`: `str | pathlib.Path` — path-like value to transform.
- `root`: `str | pathlib.Path` — containment root.
- `resolve_paths`: `bool` — resolve paths before containment testing.

## Returns

A boolean result.

## Errors and edge cases

Returns `False` for paths outside the root; neither path must exist unless resolution encounters an operating-system error.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.is_within(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`is_within.py`](../../../examples/files/is_within.py) for an executable example.

## Tests

See [`test_is_within.py`](../../../test/files/test_is_within.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.is_within."""

from klygo import files

print(files.is_within("dataset/images/a.jpg", "dataset"))
print(files.is_within("../outside.txt", "dataset"))


# Additional cases
assert files.is_within("dataset/labels/a.txt", "dataset")
assert not files.is_within("../outside.txt", "dataset")
```
