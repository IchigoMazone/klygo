# `archive.merge`

```python
archive.merge(archive_paths, output_path, overwrite=False, verbose=True)
```

Merge multiple archives into one destination archive.

## Contract

Archives with matching formats use the backend fast path. Cross-format inputs are extracted into a temporary directory and recompressed.
Archive paths accept both strings and `pathlib.Path` values. Format-specific work is delegated to the selected archive backend.

## Parameters

- `archive_paths`: `list[str or pathlib.Path]` — At least two input archives.
- `output_path`: `str or pathlib.Path` — Destination archive path.
- `overwrite`: `bool, default=False` — Allow existing output entries to be replaced.
- `verbose`: `bool, default=True` — Display archive progress.

## Returns

`None` — The destination archive is created as a side effect.

## Errors and edge cases

Raises `FileExistsError` if the output exists and overwrite is disabled.

## AI usage guidance

Prefer the public form `from klygo import archive` followed by `archive.merge(...)`. Use `klygo.files` for surrounding filesystem work. Do not import `archive.backend` directly unless implementing a new archive format.

## Example

See [`merge.py`](../../../examples/archive/merge.py) for an executable example.

## Tests

See [`test_merge.py`](../../../test/archive/test_merge.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable examples for klygo.archive.merge."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = Path(directory)
    source = make_source(root)
    first, second = root / "a.zip", root / "b.zip"
    archive.compress(source, first, verbose=False)
    archive.compress(source, second, verbose=False)

    merged = root / "merged.zip"
    archive.merge([first, second], merged, verbose=False)
    assert archive.test(merged)
    print(archive.list_files(merged))
```
