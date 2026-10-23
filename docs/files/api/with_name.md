# `files.with_name`

```python
files.with_name(value, new_name)
```

Return a Path with a replaced final component.

## Contract

Returns a new path without changing the filesystem. To rename an existing entry, use [`files.rename()`](rename.md) or `Path.rename()`.

Does not rename anything on disk.

Inputs accept strings and text-returning `os.PathLike` objects (including `pathlib.Path`), using the current operating system's path rules. Byte paths are not supported.

## Parameters

- `value`: `str | os.PathLike[str]` — path-like value to transform.
- `new_name`: `str` — replacement basename.

## Returns

A `pathlib.Path` representing the result.

## Errors and edge cases

This helper normally does not access the filesystem. Invalid argument types may raise `TypeError`.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.with_name(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`with_name.py`](../../../examples/files/with_name.py) for an executable example.

## Tests

See [`test_with_name.py`](../../../test/files/test_with_name.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.with_name."""

from klygo import files

print(files.with_name("dataset/cat.jpg", "dog.jpg"))


# Additional cases
assert files.with_name("dataset/cat.jpg", "dog.png") == files.path("dataset/dog.png")
assert files.with_name("archive.tar.gz", "backup.zip") == files.path("backup.zip")
```
