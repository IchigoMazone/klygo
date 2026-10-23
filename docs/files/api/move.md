# `files.move`

```python
files.move(source, target, overwrite=False)
```

Move a file or directory.

## Contract

Copies and publishes the destination before removing the source. It refuses overlapping source and destination paths and replaces an existing target only with `overwrite=True`. If removing the source fails, both copies may remain so the data is still recoverable.

Inputs accept strings and text-returning `os.PathLike` objects (including `pathlib.Path`), using the current operating system's path rules. Byte paths are not supported.

## Parameters

- `source`: `str | os.PathLike[str]` — source URL or path.
- `target`: `str | os.PathLike[str]` — destination path.
- `overwrite`: `bool` — permit replacing an existing target.

## Returns

A `pathlib.Path` representing the result.

## Errors and edge cases

Raises `FileNotFoundError` for a missing source, `FileExistsError` when replacement is disabled, and `ValueError` for overlapping paths.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.move(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`move.py`](../../../examples/files/move.py) for an executable example.

## Tests

See [`test_move.py`](../../../test/files/test_move.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.move."""

from tempfile import TemporaryDirectory
from klygo import files

with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = root / "source.txt"
    files.save(source, "klygo", verbose=False)
    moved = files.move(source, root / "output" / "result.txt")
    print(moved, files.exists(source))


# Additional cases
# Moving directories preserves their contents.
with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = files.mkdir(root / "before")
    files.save(source / "data.txt", "ready", verbose=False)
    target = files.move(source, root / "after")
    assert files.is_file(target / "data.txt") and not files.exists(source)
```
