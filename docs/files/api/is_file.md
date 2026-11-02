# `files.is_file`

```python
files.is_file(path)
```

Check whether a path points to a regular file.

## Contract

Returns False for missing paths and directories.

Inputs accept strings and text-returning `os.PathLike` objects (including `pathlib.Path`), using the current operating system's path rules. Byte paths are not supported.

## Parameters

- `path`: `str | os.PathLike[str]` — filesystem path to inspect or operate on.

## Returns

A boolean result.

## Errors and edge cases

This helper normally does not access the filesystem. Invalid argument types may raise `TypeError`.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.is_file(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`is_file.py`](../../../examples/files/is_file.py) for an executable example.

## Tests

See [`test_is_file.py`](../../../test/files/test_is_file.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.is_file."""

from klygo import files

print(files.is_file("README.md"))


# Additional cases
assert not files.is_file("klygo")
assert not files.is_file("missing-file")
```
