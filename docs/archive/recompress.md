# `archive.recompress`

```python
archive.recompress(source_path, target_path, compresslevel=6, overwrite=False, verbose=True)
```

Rebuild an archive with a different compression level or format.

## Contract

Members are extracted into a temporary directory and compressed with `compresslevel`.
Archive paths accept both strings and `pathlib.Path` values. Format-specific work is delegated to the selected archive backend.

## Parameters

- `source_path`: `str or pathlib.Path` — Source archive path.
- `target_path`: `str or pathlib.Path` — Destination archive path.
- `compresslevel`: `int, default=6` — Compression level, usually from 1 through 9.
- `overwrite`: `bool, default=False` — Allow existing output entries to be replaced.
- `verbose`: `bool, default=True` — Display archive progress.

## Returns

`None` — The target archive is created as a side effect.

## Errors and edge cases

Raises `FileExistsError` if the target exists and overwrite is disabled.

## AI usage guidance

Prefer the public form `from klygo import archive` followed by `archive.recompress(...)`. Use `klygo.files` for surrounding filesystem work. Do not import `archive.backend` directly unless implementing a new archive format.

## Example

See [`recompress.py`](../../examples/archive/recompress.py) for an executable example.

## Tests

See [`test_recompress.py`](../../test/archive/test_recompress.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable examples for klygo.archive.recompress."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = Path(directory)
    source = make_archive(root)
    target = root / "compact.zip"

    archive.recompress(source, target, compresslevel=9, verbose=False)
    assert archive.test(target)
    print(files.size(source), files.size(target))
```
