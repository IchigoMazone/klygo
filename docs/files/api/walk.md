# `files.walk`

```python
files.walk(path='.')
```

Walk a directory tree lazily.

## Contract

Returns the generator produced by os.walk, yielding root, directory names, and file names.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `path`: `str | pathlib.Path` — filesystem path to inspect or operate on.

## Returns

A lazy directory-tree iterator.

## Errors and edge cases

Traversal errors are raised lazily by the underlying `os.walk` iterator.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.walk(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`walk.py`](../../../examples/files/walk.py) for an executable example.

## Tests

See [`test_walk.py`](../../../test/files/test_walk.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.walk."""

from klygo import files

for root, directories, filenames in files.walk("klygo/files"):
    print(root, directories, filenames)


# Additional cases
# Materialize only when all rows are needed.
rows = list(files.walk("klygo/files"))
assert rows and all(len(row) == 3 for row in rows)
```
