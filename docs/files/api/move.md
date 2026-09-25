# `files.move`

```python
files.move(source, target, overwrite=True)
```

Move a file or directory.

## Contract

Creates the destination parent and optionally replaces an existing target.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `source`: `str | pathlib.Path` — source URL or path.
- `target`: `str | pathlib.Path` — destination path.
- `overwrite`: `bool` — permit replacing an existing target.

## Returns

A `pathlib.Path` representing the result.

## Errors and edge cases

Raises `FileNotFoundError` for a missing source and `FileExistsError` when replacement is disabled.

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
from pathlib import Path
from klygo import files

with TemporaryDirectory() as directory:
    source = Path(directory) / "source.txt"
    source.write_text("klygo", encoding="utf-8")
    moved = files.move(source, Path(directory) / "output" / "result.txt")
    print(moved, source.exists())


# Additional cases
# Moving directories preserves their contents.
with TemporaryDirectory() as directory:
    root = Path(directory)
    source = root / "before"
    source.mkdir()
    (source / "data.txt").touch()
    target = files.move(source, root / "after")
    assert (target / "data.txt").is_file() and not source.exists()
```
