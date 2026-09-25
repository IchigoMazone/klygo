# `archive.get_info`

```python
archive.get_info(archive_path)
```

Return detailed archive metadata and compression statistics.

## Contract

Archive paths accept both strings and `pathlib.Path` values. Format-specific work is delegated to the selected archive backend.

## Parameters

- `archive_path`: `str or pathlib.Path` — Archive to inspect or modify.

## Returns

`dict[str, Any]` — Format, size, ratio, encryption, and member statistics.

## Errors and edge cases

Raises `FileNotFoundError` if the archive does not exist.

## AI usage guidance

Prefer the public form `from klygo import archive` followed by `archive.get_info(...)`. Use `klygo.files` for surrounding filesystem work. Do not import `archive.backend` directly unless implementing a new archive format.

## Example

See [`get_info.py`](../../../examples/archive/get_info.py) for an executable example.

## Tests

See [`test_get_info.py`](../../../test/archive/test_get_info.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable examples for klygo.archive.get_info."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    packed = make_archive(Path(directory))
    metadata = archive.get_info(packed)
    print("format:", metadata["format"])
    print("members:", metadata["file_count"])
    print("archive size:", metadata["human_archive_size"])
    assert metadata["format"] == "zip"
```
