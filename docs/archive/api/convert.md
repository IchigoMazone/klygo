# `archive.convert`

```python
archive.convert(source_path, target_path, overwrite=False, verbose=True)
```

Convert an archive to another format.

## Contract

Members are extracted into a temporary directory and compressed using the destination backend.
Archive paths accept both strings and `pathlib.Path` values. Format-specific work is delegated to the selected archive backend.

## Parameters

- `source_path`: `str or pathlib.Path` — Source archive path.
- `target_path`: `str or pathlib.Path` — Destination archive path.
- `overwrite`: `bool, default=False` — Allow existing output entries to be replaced.
- `verbose`: `bool, default=True` — Display archive progress.

## Returns

`None` — The target archive is created as a side effect.

## Errors and edge cases

Raises `FileExistsError` if the target exists and overwrite is disabled.

## AI usage guidance

Prefer the public form `from klygo import archive` followed by `archive.convert(...)`. Use `klygo.files` for surrounding filesystem work. Do not import `archive.backend` directly unless implementing a new archive format.

## Example

See [`convert.py`](../../../examples/archive/convert.py) for an executable example.

## Tests

See [`test_convert.py`](../../../test/archive/test_convert.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable examples for klygo.archive.convert."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = Path(directory)
    source = make_archive(root)
    target = root / "converted.tar.gz"

    archive.convert(source, target, verbose=False)
    assert archive.detect_format(target) == "tar.gz"
    assert archive.test(target)
    print(target)
```
