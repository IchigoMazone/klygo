# `archive.search`

```python
archive.search(archive_path, pattern, regex=False, case_sensitive=True)
```

Search archive member names using a glob or regular expression.

## Contract

Glob matching is the default. Set `regex=True` to use a regular expression.
Archive paths accept both strings and `pathlib.Path` values. Format-specific work is delegated to the selected archive backend.

## Parameters

- `archive_path`: `str or pathlib.Path` — Archive to inspect or modify.
- `pattern`: `str` — Glob pattern or regular expression.
- `regex`: `bool, default=False` — Interpret `pattern` as a regular expression.
- `case_sensitive`: `bool, default=True` — Match letter case exactly.

## Returns

`list[str]` — Member names matching the requested pattern.

## Errors and edge cases

Backend-specific I/O, format, password, and optional-dependency errors propagate unchanged.

## AI usage guidance

Prefer the public form `from klygo import archive` followed by `archive.search(...)`. Use `klygo.files` for surrounding filesystem work. Do not import `archive.backend` directly unless implementing a new archive format.

## Example

See [`search.py`](../../../examples/archive/search.py) for an executable example.

## Tests

See [`test_search.py`](../../../test/archive/test_search.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable examples for klygo.archive.search."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    packed = make_archive(Path(directory))
    print("glob:", archive.search(packed, "*.txt"))
    print("regex:", archive.search(packed, r".*data\.json$", regex=True))
    print("case-insensitive:", archive.search(packed, "*.TXT", case_sensitive=False))
```
