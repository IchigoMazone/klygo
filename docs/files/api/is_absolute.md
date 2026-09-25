# `files.is_absolute`

```python
files.is_absolute(value)
```

Check whether a path is absolute.

## Contract

This is a pure path operation and does not check existence.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `value`: `str | pathlib.Path` — path-like value to transform.

## Returns

A boolean result.

## Errors and edge cases

This helper normally does not access the filesystem. Invalid argument types may raise `TypeError`.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.is_absolute(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`is_absolute.py`](../../../examples/files/is_absolute.py) for an executable example.

## Tests

See [`test_is_absolute.py`](../../../test/files/test_is_absolute.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.is_absolute."""

from klygo import files

print(files.is_absolute(files.resolve("README.md")))
print(files.is_absolute("README.md"))


# Additional cases
absolute = files.resolve("README.md")
assert files.is_absolute(absolute)
assert not files.is_absolute("README.md")
```
