# `archive.extract_matching`

```python
archive.extract_matching(archive_path, pattern, output_dir='.', overwrite=False, password=None)
```

Extract members matching a wildcard pattern.

## Contract

This is a convenience wrapper around `extract` with `include=pattern` and progress output disabled.
Archive paths accept both strings and `pathlib.Path` values. Format-specific work is delegated to the selected archive backend.

## Parameters

- `archive_path`: `str or pathlib.Path` — Archive to inspect or modify.
- `pattern`: `str` — Glob pattern or regular expression.
- `output_dir`: `str or pathlib.Path, default='.'` — Directory receiving extracted files or archive parts.
- `overwrite`: `bool, default=False` — Allow existing output entries to be replaced.
- `password`: `str or None, default=None` — Password for encrypted formats that support it.

## Returns

`None` — Matching members are extracted as a side effect.

## Errors and edge cases

Backend-specific I/O, format, password, and optional-dependency errors propagate unchanged.

## AI usage guidance

Prefer the public form `from klygo import archive` followed by `archive.extract_matching(...)`. Use `klygo.files` for surrounding filesystem work. Do not import `archive.backend` directly unless implementing a new archive format.

## Example

See [`extract_matching.py`](../../examples/archive/extract_matching.py) for an executable example.

## Tests

See [`test_extract_matching.py`](../../test/archive/test_extract_matching.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable examples for klygo.archive.extract_matching."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = Path(directory)
    packed = make_archive(root)

    output = root / "text-members"
    archive.extract_matching(packed, "*.txt", output)
    matches = files.find(output)
    assert len(matches) == 1
    print(matches)
```
