# `archive.extract_file`

```python
archive.extract_file(archive_path, filename, output_dir='.', overwrite=False, password=None)
```

Extract one archive member using streaming I/O.

## Contract

Only the requested member is copied to the output directory. The destination uses the member basename.
Archive paths accept both strings and `pathlib.Path` values. Format-specific work is delegated to the selected archive backend.

## Parameters

- `archive_path`: `str or pathlib.Path` — Archive to inspect or modify.
- `filename`: `str` — Exact member name stored in the archive.
- `output_dir`: `str or pathlib.Path, default='.'` — Directory receiving extracted files or archive parts.
- `overwrite`: `bool, default=False` — Allow existing output entries to be replaced.
- `password`: `str or None, default=None` — Password for encrypted formats that support it.

## Returns

`None` — The selected member is extracted as a side effect.

## Errors and edge cases

Raises `KeyError` if the member does not exist.

Raises `FileExistsError` if the destination exists and overwrite is disabled.

## AI usage guidance

Prefer the public form `from klygo import archive` followed by `archive.extract_file(...)`. Use `klygo.files` for surrounding filesystem work. Do not import `archive.backend` directly unless implementing a new archive format.

## Example

See [`extract_file.py`](../../examples/archive/extract_file.py) for an executable example.

## Tests

See [`test_extract_file.py`](../../test/archive/test_extract_file.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable examples for klygo.archive.extract_file."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = Path(directory)
    packed = make_archive(root)
    names = archive.list_files(packed)
    member = member_ending(names, "data.json")

    output = root / "single"
    archive.extract_file(packed, member, output)
    assert files.is_file(output / "data.json")
    print(output / "data.json")
```
