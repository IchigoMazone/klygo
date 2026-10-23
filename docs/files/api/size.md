# `files.size`

```python
files.size(path, human=False)
```

Calculate the size of a file or directory.

## Contract

Directory sizes include all descendant files. human=True returns a formatted string.

Inputs accept strings and text-returning `os.PathLike` objects (including `pathlib.Path`), using the current operating system's path rules. Byte paths are not supported.

## Parameters

- `path`: `str | os.PathLike[str]` — filesystem path to inspect or operate on.
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

from tempfile import TemporaryDirectory
from klygo import files

print(files.size("README.md"))
print(files.size("README.md", human=True))


# Additional cases
# Directory size is the sum of descendant files.
with TemporaryDirectory() as directory:
    root = files.path(directory)
    files.save(root / "a.txt", "123", verbose=False)
    files.save(root / "b.txt", "45", verbose=False)
    assert files.size(root) == 5
```
