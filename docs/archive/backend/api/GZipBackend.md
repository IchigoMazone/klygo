# `backend.GZipBackend`

```python
from klygo.archive.backend import GZipBackend

backend = GZipBackend()
```

Single-file GZip compression and extraction adapter (`.gz`).

## Contract

`GZipBackend` treats `.gz` files as single-stream compressed files rather than multi-member directory containers. The logical member name is inferred from the archive name by stripping `.gz`. Creation and extraction are supported, while container-level mutations (`add`, `remove`, `merge`, `split`) are intentionally disabled.

## Parameters

- `GZipBackend()` takes no constructor parameters.
- Key class attributes:
  - `format_name`: `"gz"`.
  - `capabilities`: Supports `compress=True` and option `compresslevel`.

## Returns

`GZipBackend`
    A newly constructed backend adapter for single-file GZip streams.

## Errors and edge cases

Attempting to compress a directory raises `ValueError` (use `TarBackend("tar.gz")` instead). Invoking `add`, `remove`, `merge`, or `split` raises `UnsupportedOperationError`.

## AI usage guidance

Use `GZipBackend` for standalone `.gz` compressed files. For directory archives or multi-file archives, use `tar.gz` via `klygo.archive`.

## Example

See [`GZipBackend.py`](../../../../examples/archive/backend/GZipBackend.py).

## Tests

See [`test_GZipBackend.py`](../../../../test/archive/backend/test_GZipBackend.py).

## Complete executable example

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import klygo.files as files
from klygo.archive.backend import GZipBackend

with TemporaryDirectory() as directory:
    root = Path(directory)
    source = root / "log.txt"
    files.save(source, "log entry", verbose=False)
    packed = root / "log.txt.gz"

    backend = GZipBackend()
    backend.compress(source, packed, verbose=False)
    assert backend.list_files(packed) == ["log.txt"]
    assert backend.test(packed) is True
```
