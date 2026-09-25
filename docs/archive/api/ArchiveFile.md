# `archive.ArchiveFile`

```python
archive.ArchiveFile(archive_path)
```

Object-oriented interface to a single archive.

## Contract

Format detection and backend selection happen once during construction. The object can then list, search, inspect, validate, and extract members.
Exposed attributes: `archive_path` (`pathlib.Path`) is the normalized path; `format` (`str`) is the canonical detected format; and `backend` (`ArchiveBackend`) handles format-specific operations.
Archive paths accept both strings and `pathlib.Path` values. Format-specific work is delegated to the selected archive backend.

## Parameters

- `archive_path`: `str or pathlib.Path` — Existing archive to open.

## Returns

An `ArchiveFile` object bound to the normalized archive path and selected backend.

## Errors and edge cases

Backend-specific I/O, format, password, and optional-dependency errors propagate unchanged.

## AI usage guidance

Prefer the public form `from klygo import archive` followed by `archive.ArchiveFile(...)`. Use `klygo.files` for surrounding filesystem work. Do not import `archive.backend` directly unless implementing a new archive format.

## Example

See [`ArchiveFile.py`](../../../examples/archive/ArchiveFile.py) for an executable example.

## Tests

See [`test_ArchiveFile.py`](../../../test/archive/test_ArchiveFile.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable examples for klygo.archive.ArchiveFile."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = Path(directory)
    packed = make_archive(root)
    opened = archive.ArchiveFile(packed)

    print(opened.format)
    print(opened.search("*.json"))
    output = root / "restored"
    opened.extract(output, verbose=False)
    assert files.find(output)
```
