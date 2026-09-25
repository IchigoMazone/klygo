# `archive.remove`

```python
archive.remove(archive_path, files)
```

Remove named members from an existing archive.

## Contract

Archive paths accept both strings and `pathlib.Path` values. Format-specific work is delegated to the selected archive backend.

## Parameters

- `archive_path`: `str or pathlib.Path` — Archive to inspect or modify.
- `files`: `path-like or sequence` — Filesystem inputs for `add` or member names for `remove`.

## Returns

`None` — The archive is modified as a side effect.

## Errors and edge cases

Raises `KeyError` if a requested member is absent.

## AI usage guidance

Prefer the public form `from klygo import archive` followed by `archive.remove(...)`. Use `klygo.files` for surrounding filesystem work. Do not import `archive.backend` directly unless implementing a new archive format.

## Example

See [`remove.py`](../../examples/archive/remove.py) for an executable example.

## Tests

See [`test_remove.py`](../../test/archive/test_remove.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable examples for klygo.archive.remove."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = Path(directory)
    packed = make_archive(root)
    member = member_ending(archive.list_files(packed), "alpha.txt")

    archive.remove(packed, member)
    assert member not in archive.list_files(packed)
    print("removed:", member)
```
