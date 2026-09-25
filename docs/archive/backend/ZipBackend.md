# `backend.ZipBackend`

```python
from klygo.archive.backend import ZipBackend

backend = ZipBackend()
```

Read and write ZIP archives using Python's standard library.

## Contract

`ZipBackend` is the most feature-complete built-in backend. It provides creation, extraction, streaming member reading, password decryption, member mutation (`add`, `remove`), same-format fast merging, and logical size-based splitting. All direct backend methods operate on normalized `pathlib.Path` instances.

## Parameters

- `ZipBackend()` takes no constructor parameters.
- Key class attributes:
  - `format_name`: `"zip"`.
  - `capabilities`: Declares full mutation capabilities (`compress=True`, `add=True`, `remove=True`, `merge=True`, `split=True`).
  - `COMPRESSION_METHODS`: Maps `"deflated"`, `"stored"`, `"bzip2"`, and `"lzma"` to standard `zipfile` constants.

## Returns

`ZipBackend`
    A newly constructed backend adapter for ZIP archives.

## Errors and edge cases

Raises `FileExistsError` when creating or extracting without `overwrite=True`. Raises `KeyError` when an exact requested member is missing. Rejects unsafe member paths escaping the extraction root with `ValueError`.

## AI usage guidance

Prefer using high-level `klygo.archive` functions. Instantiate `ZipBackend` directly only when low-level control or specialized ZIP operations are required.

## Example

See [`ZipBackend.py`](../../../examples/archive/backend/ZipBackend.py).

## Tests

See [`test_ZipBackend.py`](../../../test/archive/backend/test_ZipBackend.py).

## Complete executable example

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import klygo.files as files
from klygo.archive.backend import ZipBackend

with TemporaryDirectory() as directory:
    root = Path(directory)
    source = root / "sample.txt"
    files.save(source, "content", verbose=False)
    packed = root / "sample.zip"

    backend = ZipBackend()
    backend.compress(source, packed, verbose=False)
    assert "sample.txt" in backend.list_files(packed)
    assert backend.test(packed) is True
```
