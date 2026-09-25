# `archive.copy`

```python
archive.copy(source_path, target_path, overwrite=False)
```

Copy an archive without extracting or changing it.

## Contract

The implementation delegates filesystem behavior and overwrite handling to `klygo.files.copy`.
Archive paths accept both strings and `pathlib.Path` values. Format-specific work is delegated to the selected archive backend.

## Parameters

- `source_path`: `str or pathlib.Path` — Source archive path.
- `target_path`: `str or pathlib.Path` — Destination archive path.
- `overwrite`: `bool, default=False` — Allow existing output entries to be replaced.

## Returns

`None` — The destination file is created as a side effect.

## Errors and edge cases

Raises `FileNotFoundError` if the source does not exist.

Raises `FileExistsError` if the target exists and overwrite is disabled.

## AI usage guidance

Prefer the public form `from klygo import archive` followed by `archive.copy(...)`. Use `klygo.files` for surrounding filesystem work. Do not import `archive.backend` directly unless implementing a new archive format.

## Example

See [`copy.py`](../../examples/archive/copy.py) for an executable example.

## Tests

See [`test_copy.py`](../../test/archive/test_copy.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable examples for klygo.archive.copy."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = Path(directory)
    source = make_archive(root)
    target = root / "backup" / "dataset.zip"

    archive.copy(source, target)
    assert files.compare(source, target)
    print(target)
```
