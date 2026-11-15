# `archive.is_valid`

```python
archive.is_valid(archive_path, raise_exception=False)
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

Prefer the public form `from klygo import archive` followed by `archive.is_valid(...)`. Use `klygo.files` for surrounding filesystem work. Do not import `archive.backend` directly unless implementing a new archive format.

## Example

See [`is_valid.py`](../../../examples/archive/is_valid.py) for an executable example.

## Tests

See [`test_is_valid.py`](../../../test/archive/test_is_valid.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable examples for klygo.archive.test."""

from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = files.path(directory)
    packed = make_archive(root)
    assert archive.is_valid(packed)

    broken = root / "broken.zip"
    broken.write_bytes(b"PK\x03\x04broken")
    assert archive.is_valid(broken) is False
    print("valid and corrupted cases checked")
```
