# `archive.extract`

```python
archive.extract(archive_path, output_dir='.', overwrite=False, verbose=True, password=None, include=None, exclude=None)
```

Extract selected or all members from an archive.

## Contract

Member filters are applied before extraction. Backends reject path traversal entries and enforce the overwrite policy.
Archive paths accept both strings and `pathlib.Path` values. Format-specific work is delegated to the selected archive backend.

## Parameters

- `archive_path`: `str or pathlib.Path` — Archive to inspect or modify.
- `output_dir`: `str or pathlib.Path, default='.'` — Directory receiving extracted files or archive parts.
- `overwrite`: `bool, default=False` — Allow existing output entries to be replaced.
- `verbose`: `bool, default=True` — Display archive progress.
- `password`: `str or None, default=None` — Password for encrypted formats that support it.
- `include`: `str, list[str], or None, default=None` — Glob pattern or patterns to include.
- `exclude`: `str, list[str], or None, default=None` — Glob pattern or patterns to exclude.

## Returns

`None` — Members are extracted as a side effect.

## Errors and edge cases

Raises `FileNotFoundError` if the archive does not exist.

Raises `FileExistsError` if an extracted target exists and overwrite is disabled.

Raises `ValueError` if an unsafe member path is detected.

## AI usage guidance

Prefer the public form `from klygo import archive` followed by `archive.extract(...)`. Use `klygo.files` for surrounding filesystem work. Do not import `archive.backend` directly unless implementing a new archive format.

## Example

See [`extract.py`](../../../examples/archive/extract.py) for an executable example.

## Tests

See [`test_extract.py`](../../../test/archive/test_extract.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable examples for klygo.archive.extract."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = Path(directory)
    packed = make_archive(root)

    all_output = root / "all"
    archive.extract(packed, all_output, verbose=False)
    print(files.find(all_output))

    json_output = root / "json-only"
    archive.extract(packed, json_output, include="*.json", verbose=False)
    assert all(path.suffix == ".json" for path in files.find(json_output))
```
