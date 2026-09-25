# `files.copy`

```python
files.copy(source, target, overwrite=True)
```

Copy a file or directory.

## Contract

Preserves file metadata. Directory targets are replaced when overwrite=True.

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

Prefer the public form `from klygo import files` followed by `files.copy(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`copy.py`](../../../examples/files/copy.py) for an executable example.

## Tests

See [`test_copy.py`](../../../test/files/test_copy.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.copy."""

from tempfile import TemporaryDirectory
from pathlib import Path
from klygo import files

with TemporaryDirectory() as directory:
    source = Path(directory) / "source.txt"
    source.write_text("klygo", encoding="utf-8")
    copied = files.copy(source, Path(directory) / "backup" / "source.txt")
    print(copied.read_text(encoding="utf-8"))


# Additional cases
# Directories are copied recursively too.
with TemporaryDirectory() as directory:
    root = Path(directory)
    source = root / "dataset"
    source.mkdir()
    (source / "labels.txt").write_text("cat", encoding="utf-8")
    copied = files.copy(source, root / "backup")
    assert (copied / "labels.txt").read_text(encoding="utf-8") == "cat"
```
