# `backend.get_backend`

```python
from klygo.archive.backend import get_backend

backend = get_backend(path, format_hint=None)
```

Construct the backend selected by a format hint or archive path.

## Contract

`get_backend` returns a new backend instance corresponding to the detected format or the explicit `format_hint`. The `format_hint` parameter takes precedence over the path. Aliases like `"tgz"`, `"txz"`, and `"tbz2"` are automatically resolved to their canonical backend forms.

## Parameters

- `path`: `str or pathlib.Path` — Archive file path used for format detection when `format_hint` is omitted.
- `format_hint`: `str or None`, default=None — Explicit format name or alias.

## Returns

`ArchiveBackend`
    A newly instantiated backend adapter.

## Errors and edge cases

Raises `ValueError` if the format is unrecognized or no registered backend factory exists for it.

## AI usage guidance

Use `get_backend` when implementing low-level archive processing pipelines that require direct access to backend methods.

## Example

See [`get_backend.py`](../../../../examples/archive/backend/get_backend.py).

## Tests

See [`test_get_backend.py`](../../../../test/archive/backend/test_get_backend.py).

## Complete executable example

```python
from klygo.archive.backend import get_backend, ZipBackend, TarBackend

zip_backend = get_backend("archive.zip")
assert isinstance(zip_backend, ZipBackend)

tar_backend = get_backend("unused", format_hint="tgz")
assert isinstance(tar_backend, TarBackend)
assert tar_backend.format_name == "tar.gz"
```
