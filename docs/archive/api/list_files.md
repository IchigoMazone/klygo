# `archive.list_files`

```python
archive.list_files(archive_path)
```

Return the member names stored in an archive.

## Contract

Archive paths accept both strings and `pathlib.Path` values. Format-specific work is delegated to the selected archive backend.

## Parameters

- `archive_path`: `str or pathlib.Path` — Archive to inspect or modify.

## Returns

`list[str]` — Stored member names in backend order.

## Errors and edge cases

Raises `FileNotFoundError` if the archive does not exist.

## AI usage guidance

Prefer the public form `from klygo import archive` followed by `archive.list_files(...)`. Use `klygo.files` for surrounding filesystem work. Do not import `archive.backend` directly unless implementing a new archive format.

## Example

See [`list_files.py`](../../../examples/archive/list_files.py) for an executable example.

## Tests

See [`test_list_files.py`](../../../test/archive/test_list_files.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable examples for klygo.archive.list_files."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    packed = make_archive(Path(directory))
    names = archive.list_files(packed)
    assert any(name.endswith("alpha.txt") for name in names)
    assert any(name.endswith("data.json") for name in names)
    print(*names, sep="\n")
```
