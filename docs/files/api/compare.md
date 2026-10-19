# `files.compare`

```python
files.compare(path1, path2, by='content')
```

Compare two files by checksum or binary content.

## Contract

Compares file contents by default and returns early when sizes differ. `by` must be `'hash'` or `'content'`; hash mode uses SHA-256.

Inputs accept strings and text-returning `os.PathLike` objects (including `pathlib.Path`), using the current operating system's path rules. Byte paths are not supported.

## Parameters

- `path1`: `str | os.PathLike[str]` — first file to compare.
- `path2`: `str | os.PathLike[str]` — second file to compare.
- `by`: `str` — either `hash` or `content`.

## Returns

A boolean result.

## Errors and edge cases

Raises `FileNotFoundError` when either path is missing and `ValueError` for an unsupported comparison strategy.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.compare(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`compare.py`](../../../examples/files/compare.py) for an executable example.

## Tests

See [`test_compare.py`](../../../test/files/test_compare.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.compare."""

from tempfile import TemporaryDirectory
from pathlib import Path
from klygo import files

with TemporaryDirectory() as directory:
    first = Path(directory) / "a.txt"
    second = Path(directory) / "b.txt"
    first.write_text("same", encoding="utf-8")
    second.write_text("same", encoding="utf-8")
    print(files.compare(first, second, by="content"))


# Additional cases
# Different content returns False without raising.
with TemporaryDirectory() as directory:
    root = Path(directory)
    first, second = root / "a.bin", root / "b.bin"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    assert not files.compare(first, second)
    assert not files.compare(first, second, by="content")
```
