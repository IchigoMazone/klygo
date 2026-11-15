# `archive.info`

```python
archive.info(archive_path)
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

Prefer the public form `from klygo import archive` followed by `archive.info(...)`. Use `klygo.files` for surrounding filesystem work. Do not import `archive.backend` directly unless implementing a new archive format.

## Example

See [`info.py`](../../../examples/archive/info.py) for an executable example.

## Tests

See [`test_info.py`](../../../test/archive/test_info.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable examples for klygo.archive.info."""

from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    packed = make_archive(files.path(directory))
    metadata = archive.info(packed)
    print("format:", metadata["format"])
    print("members:", metadata["file_count"])
    print("archive size:", metadata["human_archive_size"])
    assert metadata["format"] == "zip"
```
