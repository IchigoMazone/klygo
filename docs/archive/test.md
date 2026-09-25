# `archive.test`

```python
archive.test(archive_path, raise_exception=False)
```

Test archive integrity.

## Contract

Corruption normally returns `False`; `raise_exception=True` converts backend failures into `ValueError`.
Archive paths accept both strings and `pathlib.Path` values. Format-specific work is delegated to the selected archive backend.

## Parameters

- `archive_path`: `str or pathlib.Path` — Archive to inspect or modify.
- `raise_exception`: `bool, default=False` — Raise `ValueError` instead of returning `False` on corruption.

## Returns

`bool` — `True` when the archive passes integrity checks.

## Errors and edge cases

Raises `ValueError` if integrity fails and `raise_exception=True`.

## AI usage guidance

Prefer the public form `from klygo import archive` followed by `archive.test(...)`. Use `klygo.files` for surrounding filesystem work. Do not import `archive.backend` directly unless implementing a new archive format.

## Example

See [`test.py`](../../examples/archive/test.py) for an executable example.

## Tests

See [`test_test.py`](../../test/archive/test_test.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable examples for klygo.archive.test."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = Path(directory)
    packed = make_archive(root)
    assert archive.test(packed)

    broken = root / "broken.zip"
    broken.write_bytes(b"PK\x03\x04broken")
    assert archive.test(broken) is False
    print("valid and corrupted cases checked")
```
