# `archive.open`

```python
archive.open(archive_path)
```

Create an `ArchiveFile` context-manager wrapper.

## Contract

Archive paths accept both strings and `pathlib.Path` values. Format-specific work is delegated to the selected archive backend.

## Parameters

- `archive_path`: `str or pathlib.Path` — Archive to inspect or modify.

## Returns

`ArchiveFile` — Context-manager wrapper for the archive.

## Errors and edge cases

Backend-specific I/O, format, password, and optional-dependency errors propagate unchanged.

## AI usage guidance

Prefer the public form `from klygo import archive` followed by `archive.open(...)`. Use `klygo.files` for surrounding filesystem work. Do not import `archive.backend` directly unless implementing a new archive format.

## Example

See [`open.py`](../../../examples/archive/open.py) for an executable example.

## Tests

See [`test_open.py`](../../../test/archive/test_open.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable examples for klygo.archive.open."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    packed = make_archive(Path(directory))
    with archive.open(packed) as opened:
        print("format:", opened.format)
        print("members:", opened.list_files())
        assert opened.test()
```
