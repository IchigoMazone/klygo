# `archive.is_archive`

```python
archive.is_archive(path)
```

Check whether a path appears to be a supported archive.

## Contract

The function is intentionally non-raising and returns `False` for missing, unsupported, or unreadable paths.
Archive paths accept both strings and `pathlib.Path` values. Format-specific work is delegated to the selected archive backend.

## Parameters

- `path`: `str or pathlib.Path` — Archive path or candidate filename.

## Returns

`bool` — Whether detection succeeds with a supported format.

## Errors and edge cases

Backend-specific I/O, format, password, and optional-dependency errors propagate unchanged.

## AI usage guidance

Prefer the public form `from klygo import archive` followed by `archive.is_archive(...)`. Use `klygo.files` for surrounding filesystem work. Do not import `archive.backend` directly unless implementing a new archive format.

## Example

See [`is_archive.py`](../../examples/archive/is_archive.py) for an executable example.

## Tests

See [`test_is_archive.py`](../../test/archive/test_is_archive.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable examples for klygo.archive.is_archive."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = Path(directory)
    packed = make_archive(root)
    plain = root / "notes.txt"
    files.save(plain, "plain text", verbose=False)

    assert archive.is_archive(packed)
    assert not archive.is_archive(plain)
    assert not archive.is_archive(root / "missing")
    print("archive detection checked")
```
