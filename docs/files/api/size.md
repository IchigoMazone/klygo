# `files.size`

```python
files.size(path, human=False)
```

Calculate the size of a file or directory.

## Contract

Directory sizes include all descendant files. human=True returns a formatted string.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `path`: `str | pathlib.Path` — filesystem path to inspect or operate on.
- `human`: `bool` — format bytes for display.

## Returns

An integer byte count, or a human-readable string.

## Errors and edge cases

Raises `FileNotFoundError` when the path does not exist.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.size(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`size.py`](../../../examples/files/size.py) for an executable example.

## Tests

See [`test_size.py`](../../../test/files/test_size.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.size."""

from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files

print(files.size("README.md"))
print(files.size("README.md", human=True))


# Additional cases
# Directory size is the sum of descendant files.
with TemporaryDirectory() as directory:
    root = Path(directory)
    (root / "a.bin").write_bytes(b"123")
    (root / "b.bin").write_bytes(b"45")
    assert files.size(root) == 5
```
