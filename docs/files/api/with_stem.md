# `files.with_stem`

```python
files.with_stem(value, new_stem)
```

Return a Path with a replaced stem.

## Contract

Preserves the final extension and does not modify the filesystem.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `value`: `str | pathlib.Path` — path-like value to transform.
- `new_stem`: `str` — replacement stem.

## Returns

A `pathlib.Path` representing the result.

## Errors and edge cases

This helper normally does not access the filesystem. Invalid argument types may raise `TypeError`.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.with_stem(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`with_stem.py`](../../../examples/files/with_stem.py) for an executable example.

## Tests

See [`test_with_stem.py`](../../../test/files/test_with_stem.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.with_stem."""

from klygo import files

print(files.with_stem("dataset/cat.jpg", "dog"))


# Additional cases
assert files.with_stem("dataset/cat.jpg", "dog") == files.path("dataset/dog.jpg")
assert files.with_stem("archive.tar.gz", "backup") == files.path("backup.gz")
```
