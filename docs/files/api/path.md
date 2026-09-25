# `files.path`

```python
files.path(value, expand_user=True)
```

Convert a string or Path to pathlib.Path.

## Contract

Expands a leading user home marker when expand_user=True.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `value`: `str | pathlib.Path` — path-like value to transform.
- `expand_user`: `bool` — expand a leading `~`.

## Returns

A `pathlib.Path` representing the result.

## Errors and edge cases

Raises `TypeError` when the value is not a string or `pathlib.Path`.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.path(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`path.py`](../../../examples/files/path.py) for an executable example.

## Tests

See [`test_path.py`](../../../test/files/test_path.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.path."""

from klygo import files

value = files.path("dataset/images")
print(value, type(value))


# Additional cases
home = files.path("~")
literal = files.path("~", expand_user=False)
assert home != literal and literal.name == "~"
```
