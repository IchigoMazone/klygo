# `archive.iter_files`

```python
archive.iter_files(archive_path)
```

Iterate over archive member names lazily.

## Contract

The backend controls iteration and avoids building an additional result list.
Archive paths accept both strings and `pathlib.Path` values. Format-specific work is delegated to the selected archive backend.

## Parameters

- `archive_path`: `str or pathlib.Path` — Archive to inspect or modify.

## Returns

`Iterator[str]` — Lazy stream of stored member names.

## Errors and edge cases

Backend-specific I/O, format, password, and optional-dependency errors propagate unchanged.

## AI usage guidance

Prefer the public form `from klygo import archive` followed by `archive.iter_files(...)`. Use `klygo.files` for surrounding filesystem work. Do not import `archive.backend` directly unless implementing a new archive format.

## Example

See [`iter_files.py`](../../examples/archive/iter_files.py) for an executable example.

## Tests

See [`test_iter_files.py`](../../test/archive/test_iter_files.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable examples for klygo.archive.iter_files."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    packed = make_archive(Path(directory))
    iterator = archive.iter_files(packed)
    assert hasattr(iterator, "__next__")
    for member in iterator:
        print(member)
```
