# `files.copy`

```python
files.copy(source, target, overwrite=False)
```

Copy a file or directory.

## Contract

Preserves file metadata. The copy is staged before publication, and an existing target is replaced only with `overwrite=True`. Source and target cannot be the same path or contain each other when a directory is involved. A directly copied symlink keeps its referent even when the destination directory changes; nested symlinks in a copied tree retain their link text.

Inputs accept strings and text-returning `os.PathLike` objects (including `pathlib.Path`), using the current operating system's path rules. Byte paths are not supported.

## Parameters

- `source`: `str | os.PathLike[str]` — source URL or path.
- `target`: `str | os.PathLike[str]` — destination path.
- `overwrite`: `bool` — permit replacing an existing target.

## Returns

A `pathlib.Path` representing the result.

## Errors and edge cases

Raises `FileNotFoundError` for a missing source, `FileExistsError` when replacement is disabled, and `ValueError` for overlapping source and target paths. If rollback fails, the error names the retained backup path.

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
