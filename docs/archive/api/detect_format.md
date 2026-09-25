# `archive.detect_format`

```python
archive.detect_format(path)
```

Detect an archive format from its name and magic bytes.

## Contract

Known compound extensions are checked first. Existing files are also inspected for ZIP, GZip, XZ, 7Z, RAR, and TAR signatures.
Archive paths accept both strings and `pathlib.Path` values. Format-specific work is delegated to the selected archive backend.

## Parameters

- `path`: `str or pathlib.Path` — Archive path or candidate filename.

## Returns

`str` — Canonical format identifier such as `zip` or `tar.gz`.

## Errors and edge cases

Raises `ValueError` if neither the extension nor file signature is supported.

## AI usage guidance

Prefer the public form `from klygo import archive` followed by `archive.detect_format(...)`. Use `klygo.files` for surrounding filesystem work. Do not import `archive.backend` directly unless implementing a new archive format.

## Example

See [`detect_format.py`](../../../examples/archive/detect_format.py) for an executable example.

## Tests

See [`test_detect_format.py`](../../../test/archive/test_detect_format.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable examples for klygo.archive.detect_format."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


assert archive.detect_format("dataset.zip") == "zip"
assert archive.detect_format("dataset.tar.gz") == "tar.gz"
assert archive.detect_format("dataset.tbz2") == "tar.bz2"

with TemporaryDirectory() as directory:
    root = Path(directory)
    packed = make_archive(root)
    renamed = files.move(packed, root / "archive.bin")
    assert archive.detect_format(renamed) == "zip"
    print("magic-byte detection:", renamed)
```
