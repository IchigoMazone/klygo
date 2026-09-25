# `backend.TarBackend`

```python
from klygo.archive.backend import TarBackend

backend = TarBackend(format_name="tar")
```

Read and write raw and compressed TAR archives (`.tar`, `.tar.gz`, `.tar.xz`, `.tar.bz2`).

## Contract

`TarBackend` manages TAR containers across multiple compression formats. Adding to compressed archives rebuilds the container atomically to preserve archive integrity. Extraction enforces path containment checks and Python's `data` filter where available.

## Parameters

- `format_name`: `{'tar', 'tar.gz', 'tar.xz', 'tar.bz2'}`, default='tar' — TAR variant to manage.
- Key class attributes:
  - `format_name`: Canonical selected variant.
  - `capabilities`: Compressed variants support `compresslevel` (1–9), while uncompressed TAR rejects changed compression levels.

## Returns

`TarBackend`
    A newly constructed backend adapter bound to the requested TAR variant.

## Errors and edge cases

Raises `ValueError` if `format_name` is unsupported or an unsafe member path is detected. Raises `FileExistsError` if the output already exists and `overwrite=False`. Raises `KeyError` if a requested member does not exist.

## AI usage guidance

Prefer using high-level functions like `archive.compress("data", "data.tar.gz")`. Use `TarBackend` directly when explicitly constructing or testing TAR adapters.

## Example

See [`TarBackend.py`](../../../../examples/archive/backend/TarBackend.py).

## Tests

See [`test_TarBackend.py`](../../../../test/archive/backend/test_TarBackend.py).

## Complete executable example

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import klygo.files as files
from klygo.archive.backend import TarBackend

with TemporaryDirectory() as directory:
    root = Path(directory)
    source = root / "file.txt"
    files.save(source, "tar test", verbose=False)
    packed = root / "bundle.tar.gz"

    backend = TarBackend("tar.gz")
    backend.compress(source, packed, verbose=False)
    assert "file.txt" in backend.list_files(packed)
    assert backend.test(packed) is True
```
