# `backend.UnsupportedOperationError`

```python
from klygo.archive.backend import UnsupportedOperationError

raise UnsupportedOperationError(format_name, operation)
```

Report an operation that an archive format backend cannot perform.

## Contract

`UnsupportedOperationError` subclasses standard Python `NotImplementedError` so existing exception handlers catching `NotImplementedError` remain compatible. It records the canonical format name and the rejected operation name as instance attributes.

## Parameters

- `format_name`: `str` — Canonical format identifier (e.g. `"gz"`, `"rar"`).
- `operation`: `str` — Requested operation name (e.g. `"add"`, `"compress"`).

## Returns

`UnsupportedOperationError`
    Exception instance representing an unsupported archive operation.

## Errors and edge cases

This exception is raised when `ArchiveBackend.require_operation()` is called with an operation that the backend's `capabilities` flag marks as `False`.

## AI usage guidance

Catch `UnsupportedOperationError` when implementing cross-format fallbacks (for example, falling back from in-place archive mutation to temporary-directory recompression).

## Example

See [`UnsupportedOperationError.py`](../../../../examples/archive/backend/UnsupportedOperationError.py).

## Tests

See [`test_UnsupportedOperationError.py`](../../../../test/archive/backend/test_UnsupportedOperationError.py).

## Complete executable example

```python
from klygo.archive.backend import UnsupportedOperationError, GZipBackend

backend = GZipBackend()
try:
    backend.require_operation("add")
except UnsupportedOperationError as exc:
    assert exc.format_name == "gz"
    assert exc.operation == "add"
    assert isinstance(exc, NotImplementedError)
```
