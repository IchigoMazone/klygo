# `archive.compare`

```python
archive.compare(archive1, archive2)
```

Compare the member-name sets of two archives.

## Contract

File contents are not compared; this function compares normalized member-name sets.
Archive paths accept both strings and `pathlib.Path` values. Format-specific work is delegated to the selected archive backend.

## Parameters

- `archive1`: `str or pathlib.Path` — First archive.
- `archive2`: `str or pathlib.Path` — Second archive.

## Returns

`dict[str, list[str]]` — Added, removed, and common member names.

## Errors and edge cases

Backend-specific I/O, format, password, and optional-dependency errors propagate unchanged.

## AI usage guidance

Prefer the public form `from klygo import archive` followed by `archive.compare(...)`. Use `klygo.files` for surrounding filesystem work. Do not import `archive.backend` directly unless implementing a new archive format.

## Example

See [`compare.py`](../../../examples/archive/compare.py) for an executable example.

## Tests

See [`test_compare.py`](../../../test/archive/test_compare.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable examples for klygo.archive.compare."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = Path(directory)
    source = make_source(root)
    first, second = root / "v1.zip", root / "v2.zip"
    archive.compress(source, first, verbose=False)
    archive.compress(source, second, verbose=False)
    extra = root / "extra.txt"
    files.save(extra, "new", verbose=False)
    archive.add(second, extra, verbose=False)

    difference = archive.compare(first, second)
    assert difference["added_files"] == ["extra.txt"]
    print(difference)
```
