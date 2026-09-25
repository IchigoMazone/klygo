# `backend.RarBackend`

```python
from klygo.archive.backend import RarBackend

backend = RarBackend()
```

Inspect and extract RAR archives (`.rar`) via the optional `rarfile` library.

## Contract

`RarBackend` is a read-only adapter. It supports listing, metadata inspection, integrity checking, and extraction (with password support where encrypted). Creation and mutating operations are disabled by design. The class can be imported and inspected without `rarfile` installed; operations raise an actionable `ImportError` if `rarfile` is absent.

## Parameters

- `RarBackend()` takes no constructor parameters.
- Key class attributes:
  - `format_name`: `"rar"`.
  - `capabilities`: Read-only (`compress=False`, `add=False`, `remove=False`, `merge=False`, `split=False`), with password extraction support.

## Returns

`RarBackend`
    A newly constructed read-only backend adapter for RAR archives.

## Errors and edge cases

Attempting `compress`, `add`, `remove`, `merge`, or `split` raises `UnsupportedOperationError`. Raises `ImportError` when `rarfile` is missing during extraction.

## AI usage guidance

Use `RarBackend` solely for unpacking existing RAR archives. For creating new archives, choose ZIP or TAR.GZ.

## Example

See [`RarBackend.py`](../../../examples/archive/backend/RarBackend.py).

## Tests

See [`test_RarBackend.py`](../../../test/archive/backend/test_RarBackend.py).

## Complete executable example

```python
from klygo.archive.backend import RarBackend

backend = RarBackend()
assert backend.format_name == "rar"
assert backend.capabilities.compress is False
assert "password" in backend.capabilities.extract_options
```
