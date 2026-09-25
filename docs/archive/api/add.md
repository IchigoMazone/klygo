# `archive.add`

```python
archive.add(archive_path, files, verbose=True, on_conflict='rename')
```

Add files or directories to an existing archive.

## Contract

Archive paths accept both strings and `pathlib.Path` values. Format-specific work is delegated to the selected archive backend.

## Parameters

- `archive_path`: `str or pathlib.Path` — Archive to inspect or modify.
- `files`: `path-like or sequence` — Filesystem inputs for `add` or member names for `remove`.
- `verbose`: `bool, default=True` — Display archive progress.
- `on_conflict`: `{'rename', 'overwrite', 'skip'}, default='rename'` — Strategy used when an added member name already exists.

## Returns

`None` — The archive is modified as a side effect.

## Errors and edge cases

Raises `FileNotFoundError` if an input path does not exist.

## AI usage guidance

Prefer the public form `from klygo import archive` followed by `archive.add(...)`. Use `klygo.files` for surrounding filesystem work. Do not import `archive.backend` directly unless implementing a new archive format.

## Example

See [`add.py`](../../../examples/archive/add.py) for an executable example.

## Tests

See [`test_add.py`](../../../test/archive/test_add.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable examples for klygo.archive.add."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = Path(directory)
    packed = make_archive(root)
    extra = root / "extra.txt"
    files.save(extra, "extra", verbose=False)

    archive.add(packed, extra, verbose=False)
    archive.add(packed, extra, on_conflict="rename", verbose=False)
    names = archive.list_files(packed)
    assert "extra.txt" in names
    assert any("extra_dup" in name for name in names)
    print(names)
```
