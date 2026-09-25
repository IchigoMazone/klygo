# `archive.verify`

```python
archive.verify(archive_path)
```

Build a high-level archive verification report.

## Contract

The report combines integrity status with format, member count, size, and encryption metadata.
Archive paths accept both strings and `pathlib.Path` values. Format-specific work is delegated to the selected archive backend.

## Parameters

- `archive_path`: `str or pathlib.Path` — Archive to inspect or modify.

## Returns

`dict[str, Any]` — Summary containing validity and archive metadata.

## Errors and edge cases

Backend-specific I/O, format, password, and optional-dependency errors propagate unchanged.

## AI usage guidance

Prefer the public form `from klygo import archive` followed by `archive.verify(...)`. Use `klygo.files` for surrounding filesystem work. Do not import `archive.backend` directly unless implementing a new archive format.

## Example

See [`verify.py`](../../examples/archive/verify.py) for an executable example.

## Tests

See [`test_verify.py`](../../test/archive/test_verify.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable examples for klygo.archive.verify."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    packed = make_archive(Path(directory))
    report = archive.verify(packed)
    assert report["valid"] is True
    print(report)
```
