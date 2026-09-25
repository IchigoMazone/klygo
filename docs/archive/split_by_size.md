# `archive.split_by_size`

```python
archive.split_by_size(archive_path, size, output_dir='.', overwrite=False, verbose=True)
```

Split an archive into size-limited archive parts.

## Contract

Archive paths accept both strings and `pathlib.Path` values. Format-specific work is delegated to the selected archive backend.

## Parameters

- `archive_path`: `str or pathlib.Path` — Archive to inspect or modify.
- `size`: `int or float` — Maximum size of each part in megabytes.
- `output_dir`: `str or pathlib.Path, default='.'` — Directory receiving extracted files or archive parts.
- `overwrite`: `bool, default=False` — Allow existing output entries to be replaced.
- `verbose`: `bool, default=True` — Display archive progress.

## Returns

`list[str]` — Paths to the generated archive parts.

## Errors and edge cases

Raises `ValueError` if `size` is not positive or the backend cannot split.

## AI usage guidance

Prefer the public form `from klygo import archive` followed by `archive.split_by_size(...)`. Use `klygo.files` for surrounding filesystem work. Do not import `archive.backend` directly unless implementing a new archive format.

## Example

See [`split_by_size.py`](../../examples/archive/split_by_size.py) for an executable example.

## Tests

See [`test_split_by_size.py`](../../test/archive/test_split_by_size.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable examples for klygo.archive.split_by_size."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = Path(directory)
    packed = make_archive(root)
    parts = archive.split_by_size(packed, 0.00001, root / "parts", verbose=False)
    assert parts and all(files.is_file(part) for part in parts)
    print(parts)
```
