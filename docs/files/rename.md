# `files.rename`

```python
files.rename(path, new_name_or_path, overwrite=False)
```

Rename a filesystem entry or move it to an explicit path.

## Contract

A single path component is interpreted relative to the source parent.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `path`: `str | pathlib.Path` — filesystem path to inspect or operate on.
- `new_name_or_path`: `str | pathlib.Path` — basename or explicit destination.
- `overwrite`: `bool` — permit replacing an existing target.

## Returns

A `pathlib.Path` representing the result.

## Errors and edge cases

Raises `FileNotFoundError` for a missing source and `FileExistsError` when replacement is disabled.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.rename(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`rename.py`](../../examples/files/rename.py) for an executable example.

## Tests

See [`test_rename.py`](../../test/files/test_rename.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.rename."""

from tempfile import TemporaryDirectory
from pathlib import Path
from klygo import files

with TemporaryDirectory() as directory:
    source = Path(directory) / "draft.txt"
    source.write_text("done", encoding="utf-8")
    renamed = files.rename(source, "final.txt")
    print(renamed)


# Additional cases
# A full destination path can move and rename in one operation.
with TemporaryDirectory() as directory:
    root = Path(directory)
    source = root / "a.txt"
    source.touch()
    destination = root / "nested" / "b.txt"
    destination.parent.mkdir()
    assert files.rename(source, destination) == destination
```
