# `files.as_posix`

```python
files.as_posix(path)
```

Represent a path with forward-slash separators.

## Contract

Accepts a string or text-returning `os.PathLike[str]`. Does not access the filesystem or resolve the path.

## Parameters

- `path`: `str | os.PathLike[str]` — path to represent.

## Returns

A `str` with POSIX separators.

## Errors and edge cases

Raises `TypeError` for unsupported path inputs. Relative paths remain relative.

## AI usage guidance

Use `files.as_posix(...)` when another Klygo module needs portable path text.

## Example

See [`as_posix.py`](../../../examples/files/as_posix.py).

## Tests

See [`test_as_posix.py`](../../../test/files/test_as_posix.py).

## Complete executable example

```python
"""Executable example for klygo.files.as_posix."""

from klygo import files

print(files.as_posix("dataset/images/cat.jpg"))
```
