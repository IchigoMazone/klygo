# `files.is_symlink`

```python
files.is_symlink(path)
```

Check whether a directory entry is a symbolic link, including a broken link.

## Contract

Accepts a string or text-returning `os.PathLike[str]` and inspects the entry itself.

## Parameters

- `path`: `str | os.PathLike[str]` — path to inspect.

## Returns

A `bool`; false for a regular file, directory, or missing path.

## Errors and edge cases

Filesystem access errors may raise `OSError`. A broken symbolic link still returns true.

## AI usage guidance

Use `files.is_symlink(...)` for symlink checks in other Klygo modules.

## Example

See [`is_symlink.py`](../../../examples/files/is_symlink.py).

## Tests

See [`test_is_symlink.py`](../../../test/files/test_is_symlink.py).

## Complete executable example

```python
"""Executable example for klygo.files.is_symlink."""

from tempfile import TemporaryDirectory

from klygo import files

with TemporaryDirectory() as directory:
    item = files.join(directory, "item.txt")
    files.save(item, "hello", overwrite=True, verbose=False)
    print(files.is_symlink(item))  # False for a regular file.
```
